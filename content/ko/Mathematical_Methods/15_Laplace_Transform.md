# 15. 라플라스 변환 (Laplace Transform)

## 학습 목표

- 라플라스 변환의 정의와 존재 조건을 이해하고, 수렴 영역의 개념을 설명할 수 있다
- 기본 함수들의 라플라스 변환을 유도하고, 변환표를 활용하여 복잡한 함수의 변환을 계산할 수 있다
- 이동 정리, 미분 성질, 합성곱 정리 등 라플라스 변환의 주요 성질을 증명하고 적용할 수 있다
- 부분분수 분해와 역 라플라스 변환을 이용하여 상미분방정식의 초기값 문제를 체계적으로 풀 수 있다
- 전달함수를 이용한 선형 시스템 해석과 안정성 판별의 기본 원리를 이해한다
- RLC 회로, 감쇠 진동 등 물리·공학 문제에 라플라스 변환을 적용할 수 있다

> **물리학과 공학에서의 중요성**: 라플라스 변환은 미분방정식을 대수방정식으로 변환하여 초기값 문제를 체계적으로 풀 수 있게 해주는 핵심 도구이다. 회로 해석, 제어 공학, 신호 처리, 기계 진동, 열전도 등 거의 모든 공학 분야에서 필수적이며, 푸리에 변환의 일반화로서 과도 응답(transient response) 해석에 특히 강력한 위력을 발휘한다.

---

## 1. 라플라스 변환의 정의와 존재 조건

### 1.1 정의

함수 $f(t)$ ($t \geq 0$)의 **라플라스 변환**은 다음과 같이 정의된다:

$$F(s) = \mathcal{L}\{f(t)\} = \int_0^\infty f(t) e^{-st} \, dt$$

여기서 $s = \sigma + i\omega$는 복소 변수이다. 이 적분이 수렴하는 $s$의 영역에서 $F(s)$가 정의된다.

직관적으로, 라플라스 변환은 시간 영역(time domain)의 함수 $f(t)$를 복소 주파수 영역(complex frequency domain)의 함수 $F(s)$로 변환한다. 실수부 $\sigma$는 지수적 감쇠/증가를, 허수부 $\omega$는 진동을 나타낸다.

### 1.2 존재 조건

라플라스 변환이 존재하기 위한 **충분조건**:

1. **구간별 연속(piecewise continuous)**: $f(t)$가 모든 유한 구간 $[0, T]$에서 유한 개의 불연속점을 제외하면 연속
2. **지수적 차수(exponential order)**: 적당한 상수 $M > 0$, $a \in \mathbb{R}$, $T > 0$이 존재하여

$$|f(t)| \leq M e^{at}, \quad t > T$$

이 조건이 만족되면 $\text{Re}(s) > a$인 모든 $s$에서 적분이 수렴한다.

### 1.3 수렴 영역 (Region of Convergence)

수렴 영역(ROC)은 라플라스 적분이 수렴하는 복소 평면의 영역이다:

$$\text{ROC} = \{ s \in \mathbb{C} \mid \text{Re}(s) > \sigma_c \}$$

여기서 $\sigma_c$를 **수렴 횡좌표(abscissa of convergence)**라 한다. 예를 들어:
- $f(t) = 1$: $\sigma_c = 0$ (즉 $\text{Re}(s) > 0$)
- $f(t) = e^{at}$: $\sigma_c = a$ (즉 $\text{Re}(s) > a$)
- $f(t) = e^{-t}\sin t$: $\sigma_c = -1$

```python
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# SymPy를 이용한 라플라스 변환 계산
t, s = sp.symbols('t s', positive=True)

# 기본 함수들의 라플라스 변환
funcs = {
    '1': 1,
    't': t,
    't^2': t**2,
    'exp(-2t)': sp.exp(-2*t),
    'sin(3t)': sp.sin(3*t),
    'cos(3t)': sp.cos(3*t),
}

print("=" * 50)
print("기본 함수의 라플라스 변환")
print("=" * 50)
for name, f in funcs.items():
    F = sp.laplace_transform(f, t, s, noconds=True)
    print(f"L{{{name}}} = {F}")
```

### 1.4 푸리에 변환과의 관계

라플라스 변환은 푸리에 변환의 일반화이다. $s = i\omega$로 놓으면 (그리고 $f(t) = 0$ for $t < 0$이면):

$$F(i\omega) = \int_0^\infty f(t) e^{-i\omega t} \, dt$$

이는 단측(one-sided) 푸리에 변환과 동일하다. 라플라스 변환은 $e^{-\sigma t}$라는 수렴 인자를 추가로 곱함으로써, 푸리에 변환이 존재하지 않는 함수(예: $f(t) = e^{2t}$)에도 적용 가능하다.

---

## 2. 기본 함수의 라플라스 변환

### 2.1 변환표

아래 표는 가장 중요한 라플라스 변환 쌍을 정리한 것이다:

| $f(t)$ ($t \geq 0$) | $F(s) = \mathcal{L}\{f(t)\}$ | 수렴 조건 |
|---|---|---|
| $1$ | $\dfrac{1}{s}$ | $\text{Re}(s) > 0$ |
| $t^n$ ($n = 0, 1, 2, \ldots$) | $\dfrac{n!}{s^{n+1}}$ | $\text{Re}(s) > 0$ |
| $e^{at}$ | $\dfrac{1}{s - a}$ | $\text{Re}(s) > a$ |
| $\sin(bt)$ | $\dfrac{b}{s^2 + b^2}$ | $\text{Re}(s) > 0$ |
| $\cos(bt)$ | $\dfrac{s}{s^2 + b^2}$ | $\text{Re}(s) > 0$ |
| $\sinh(bt)$ | $\dfrac{b}{s^2 - b^2}$ | $\text{Re}(s) > |b|$ |
| $\cosh(bt)$ | $\dfrac{s}{s^2 - b^2}$ | $\text{Re}(s) > |b|$ |
| $t^n e^{at}$ | $\dfrac{n!}{(s-a)^{n+1}}$ | $\text{Re}(s) > a$ |
| $e^{at}\sin(bt)$ | $\dfrac{b}{(s-a)^2 + b^2}$ | $\text{Re}(s) > a$ |
| $e^{at}\cos(bt)$ | $\dfrac{s-a}{(s-a)^2 + b^2}$ | $\text{Re}(s) > a$ |

**유도 예시**: $\mathcal{L}\{e^{at}\}$를 직접 계산하면

$$\int_0^\infty e^{at} e^{-st} \, dt = \int_0^\infty e^{-(s-a)t} \, dt = \left[ \frac{e^{-(s-a)t}}{-(s-a)} \right]_0^\infty = \frac{1}{s-a}$$

단, $\text{Re}(s) > a$이어야 $t \to \infty$에서 지수가 0으로 수렴한다.

### 2.2 단위 계단 함수 (Heaviside Function)

**헤비사이드 함수**의 정의:

$$u(t - a) = \begin{cases} 0, & t < a \\ 1, & t \geq a \end{cases}$$

라플라스 변환:

$$\mathcal{L}\{u(t-a)\} = \int_a^\infty e^{-st} \, dt = \frac{e^{-as}}{s}, \quad a \geq 0$$

### 2.3 디랙 델타 함수

$\delta(t - a)$는 $t = a$에서 단위 충격(impulse)을 나타내며:

$$\mathcal{L}\{\delta(t - a)\} = \int_0^\infty \delta(t - a) e^{-st} \, dt = e^{-as}, \quad a \geq 0$$

특히 $a = 0$이면 $\mathcal{L}\{\delta(t)\} = 1$이다. 이는 충격 입력의 라플라스 변환이 상수 1임을 의미한다.

```python
import sympy as sp

t, s, a = sp.symbols('t s a', positive=True)

# 단위 계단 함수의 라플라스 변환
heaviside_transform = sp.laplace_transform(sp.Heaviside(t - a), t, s, noconds=True)
print(f"L{{u(t-a)}} = {heaviside_transform}")

# 디랙 델타 함수의 라플라스 변환
delta_transform = sp.laplace_transform(sp.DiracDelta(t - a), t, s, noconds=True)
print(f"L{{delta(t-a)}} = {delta_transform}")

# 기본 변환 쌍 수치 검증: L{sin(3t)} = 3/(s^2+9)
import numpy as np
from scipy import integrate

def numerical_laplace(f, s_val, upper=50):
    """라플라스 변환의 수치 계산"""
    integrand = lambda tau: f(tau) * np.exp(-s_val * tau)
    result, _ = integrate.quad(integrand, 0, upper)
    return result

s_test = 2.0
# 수치 결과
numerical = numerical_laplace(lambda tau: np.sin(3*tau), s_test)
# 해석적 결과: 3/(s^2+9)
analytical = 3 / (s_test**2 + 9)

print(f"\nL{{sin(3t)}} 검증 (s={s_test}):")
print(f"  수치 적분: {numerical:.6f}")
print(f"  해석적:    {analytical:.6f}")
print(f"  오차:      {abs(numerical - analytical):.2e}")
```

---

## 3. 라플라스 변환의 성질

### 3.1 선형성

라플라스 변환은 **선형 연산자**이다:

$$\mathcal{L}\{\alpha f(t) + \beta g(t)\} = \alpha F(s) + \beta G(s)$$

이는 적분의 선형성으로부터 직접 따라온다.

### 3.2 제1이동 정리 (s-이동)

$\mathcal{L}\{f(t)\} = F(s)$이면:

$$\mathcal{L}\{e^{at}f(t)\} = F(s - a)$$

**증명**:

$$\mathcal{L}\{e^{at}f(t)\} = \int_0^\infty e^{at}f(t)e^{-st} \, dt = \int_0^\infty f(t)e^{-(s-a)t} \, dt = F(s-a)$$

**응용 예**: $\mathcal{L}\{e^{-2t}\cos(3t)\}$를 구하려면

$$\mathcal{L}\{\cos(3t)\} = \frac{s}{s^2 + 9}$$

에서 $s$를 $s + 2$로 교체하면:

$$\mathcal{L}\{e^{-2t}\cos(3t)\} = \frac{s+2}{(s+2)^2 + 9}$$

### 3.3 제2이동 정리 (t-이동)

$\mathcal{L}\{f(t)\} = F(s)$이면:

$$\mathcal{L}\{f(t-a)\,u(t-a)\} = e^{-as}F(s), \quad a > 0$$

**증명**: 치환 $\tau = t - a$를 사용하면

$$\int_0^\infty f(t-a)\,u(t-a)\,e^{-st} \, dt = \int_a^\infty f(t-a)\,e^{-st} \, dt = \int_0^\infty f(\tau)\,e^{-s(\tau+a)} \, d\tau = e^{-as}F(s)$$

이 정리는 시간 지연된 신호의 변환에 핵심적이다.

### 3.4 미분 성질

라플라스 변환의 가장 강력한 성질은 **미분을 대수적 연산으로 바꿔준다**는 것이다:

$$\mathcal{L}\{f'(t)\} = sF(s) - f(0)$$

$$\mathcal{L}\{f''(t)\} = s^2 F(s) - sf(0) - f'(0)$$

일반적으로 $n$계 도함수의 변환은:

$$\mathcal{L}\{f^{(n)}(t)\} = s^n F(s) - s^{n-1}f(0) - s^{n-2}f'(0) - \cdots - f^{(n-1)}(0)$$

**증명** (1계): 부분 적분을 적용하면

$$\int_0^\infty f'(t)e^{-st} \, dt = \left[f(t)e^{-st}\right]_0^\infty + s\int_0^\infty f(t)e^{-st} \, dt = -f(0) + sF(s)$$

### 3.5 적분 성질

$$\mathcal{L}\left\{\int_0^t f(\tau) \, d\tau\right\} = \frac{F(s)}{s}$$

시간 영역에서의 적분은 $s$-영역에서 $s$로 나누는 것에 대응한다.

### 3.6 합성곱 정리 (Convolution Theorem)

두 함수의 **합성곱**(convolution)은:

$$(f * g)(t) = \int_0^t f(\tau) \, g(t - \tau) \, d\tau$$

**합성곱 정리**:

$$\mathcal{L}\{f * g\} = F(s) \cdot G(s)$$

즉, 시간 영역에서의 합성곱은 $s$-영역에서의 곱셈에 대응한다.

**증명**: 적분 순서 교환을 이용하면

$$\mathcal{L}\{f * g\} = \int_0^\infty \left(\int_0^t f(\tau)g(t-\tau) \, d\tau \right) e^{-st} \, dt$$

$u = t - \tau$로 치환하고 적분 순서를 교환하면 $F(s) \cdot G(s)$가 된다.

### 3.7 초기값 정리와 최종값 정리

**초기값 정리**: $f(0^+) = \lim_{s \to \infty} sF(s)$

**최종값 정리**: $\lim_{t \to \infty} f(t) = \lim_{s \to 0} sF(s)$

단, 최종값 정리는 $sF(s)$의 모든 극점이 복소 평면의 왼쪽 반평면에 있을 때(즉, 시스템이 안정할 때)만 유효하다.

```python
import sympy as sp

t, s = sp.symbols('t s')
a_sym = sp.Symbol('a', positive=True)

# 제1이동 정리 검증: L{e^(-2t)*cos(3t)}
f1 = sp.exp(-2*t) * sp.cos(3*t)
F1_direct = sp.laplace_transform(f1, t, s, noconds=True)
F1_shift = (s + 2) / ((s + 2)**2 + 9)  # s-이동 적용
print("=== 제1이동 정리 검증 ===")
print(f"직접 변환:  {F1_direct}")
print(f"이동 정리:  {F1_shift}")
print(f"동일 여부:  {sp.simplify(F1_direct - F1_shift) == 0}")

# 미분 성질 검증: L{f'(t)} = sF(s) - f(0)
# f(t) = t*exp(-t), f(0) = 0
f2 = t * sp.exp(-t)
f2_prime = sp.diff(f2, t)  # (1-t)*exp(-t)

F2 = sp.laplace_transform(f2, t, s, noconds=True)  # 1/(s+1)^2
F2_prime_direct = sp.laplace_transform(f2_prime, t, s, noconds=True)
F2_prime_property = s * F2 - 0  # f(0) = 0

print("\n=== 미분 성질 검증 ===")
print(f"L{{f'(t)}} 직접:  {F2_prime_direct}")
print(f"sF(s) - f(0):    {sp.simplify(F2_prime_property)}")

# 합성곱 정리 검증: L{sin(t) * sin(t)} = [1/(s^2+1)]^2
conv_result = sp.laplace_transform(
    sp.integrate(sp.sin(t - sp.Symbol('tau')) * sp.sin(sp.Symbol('tau')),
                 (sp.Symbol('tau'), 0, t)), t, s, noconds=True
)
product_result = 1 / (s**2 + 1)**2
print("\n=== 합성곱 정리 ===")
print(f"F(s)*G(s) = 1/(s^2+1)^2 = {product_result}")
```

---

## 4. 역 라플라스 변환

### 4.1 부분분수 분해 (Partial Fraction Decomposition)

역 라플라스 변환의 핵심 기법은 $F(s)$를 부분분수로 분해한 후 변환표를 역으로 적용하는 것이다.

**경우 1**: 서로 다른 실수 근

$$\frac{P(s)}{(s-a_1)(s-a_2)\cdots(s-a_n)} = \frac{A_1}{s-a_1} + \frac{A_2}{s-a_2} + \cdots + \frac{A_n}{s-a_n}$$

**경우 2**: 중근

$$\frac{P(s)}{(s-a)^n} = \frac{A_1}{s-a} + \frac{A_2}{(s-a)^2} + \cdots + \frac{A_n}{(s-a)^n}$$

**경우 3**: 복소 켤레근

$$\frac{P(s)}{s^2 + bs + c} = \frac{As + B}{s^2 + bs + c}$$

이후 완전제곱식으로 변환하여 $\sin$, $\cos$ 변환 쌍을 적용한다.

### 4.2 헤비사이드 은폐법 (Cover-up Method)

서로 다른 1차 인수의 경우, 계수를 빠르게 구할 수 있다. $F(s)$의 분모가 $(s - a_k)$인 항의 계수:

$$A_k = \left[(s - a_k) F(s)\right]_{s = a_k}$$

**예시**: $F(s) = \dfrac{3s + 2}{(s+1)(s-2)}$의 역 변환을 구하면

$$A_1 = \left[\frac{3s+2}{s-2}\right]_{s=-1} = \frac{-1}{-3} = \frac{1}{3}$$

$$A_2 = \left[\frac{3s+2}{s+1}\right]_{s=2} = \frac{8}{3}$$

따라서 $f(t) = \frac{1}{3}e^{-t} + \frac{8}{3}e^{2t}$

### 4.3 브롬위치 적분 (Bromwich Integral)

역 라플라스 변환의 엄밀한 공식은 복소 적분으로 주어진다:

$$f(t) = \mathcal{L}^{-1}\{F(s)\} = \frac{1}{2\pi i} \int_{\gamma - i\infty}^{\gamma + i\infty} F(s) e^{st} \, ds$$

여기서 $\gamma$는 $F(s)$의 모든 특이점의 실수부보다 큰 실수이다. 이 적분은 유수 정리를 이용하여 계산할 수 있으며, 이는 12장에서 배운 복소해석의 직접적인 응용이다.

```python
import sympy as sp

s, t = sp.symbols('s t')

print("=== 부분분수 분해와 역 라플라스 변환 ===\n")

# 예제 1: 서로 다른 실수 근
F1 = (3*s + 2) / ((s + 1) * (s - 2))
print(f"F(s) = {F1}")
pf1 = sp.apart(F1, s)
print(f"부분분수: {pf1}")
f1 = sp.inverse_laplace_transform(F1, s, t)
print(f"f(t) = {f1}\n")

# 예제 2: 중근
F2 = (2*s + 3) / (s + 1)**2
print(f"F(s) = {F2}")
pf2 = sp.apart(F2, s)
print(f"부분분수: {pf2}")
f2 = sp.inverse_laplace_transform(F2, s, t)
print(f"f(t) = {f2}\n")

# 예제 3: 복소 켤레근
F3 = (s + 3) / (s**2 + 2*s + 5)
print(f"F(s) = {F3}")
# 완전제곱식: (s+1)^2 + 4 -> e^(-t)cos(2t) + e^(-t)sin(2t)
f3 = sp.inverse_laplace_transform(F3, s, t)
print(f"f(t) = {f3}\n")

# 예제 4: 고차 분모
F4 = 1 / (s * (s**2 + 4))
print(f"F(s) = {F4}")
pf4 = sp.apart(F4, s)
print(f"부분분수: {pf4}")
f4 = sp.inverse_laplace_transform(F4, s, t)
print(f"f(t) = {f4}")
```

---

## 5. 상미분방정식 풀이

### 5.1 풀이 절차

라플라스 변환을 이용한 ODE 풀이의 일반적 절차:

1. ODE의 양변에 라플라스 변환을 적용한다
2. 미분 성질을 이용하여 초기 조건을 대입한다
3. $Y(s)$에 대해 대수적으로 정리한다
4. 역 라플라스 변환으로 $y(t)$를 구한다

### 5.2 2계 상수 계수 ODE

**예제**: 다음 초기값 문제를 풀어라.

$$y'' + 3y' + 2y = 0, \quad y(0) = 1, \quad y'(0) = 0$$

**풀이**:

양변에 라플라스 변환을 적용하면:

$$[s^2 Y(s) - sy(0) - y'(0)] + 3[sY(s) - y(0)] + 2Y(s) = 0$$

초기 조건 대입:

$$s^2 Y - s + 3sY - 3 + 2Y = 0$$

$Y(s)$로 정리:

$$Y(s)(s^2 + 3s + 2) = s + 3$$

$$Y(s) = \frac{s + 3}{s^2 + 3s + 2} = \frac{s + 3}{(s+1)(s+2)}$$

부분분수 분해: $\frac{s+3}{(s+1)(s+2)} = \frac{2}{s+1} - \frac{1}{s+2}$

역 변환:

$$y(t) = 2e^{-t} - e^{-2t}$$

### 5.3 비제차 ODE

**예제**: $y'' + y = \sin(2t)$, $y(0) = 0$, $y'(0) = 0$

양변에 라플라스 변환:

$$s^2 Y(s) + Y(s) = \frac{2}{s^2 + 4}$$

$$Y(s) = \frac{2}{(s^2 + 1)(s^2 + 4)}$$

부분분수 분해:

$$\frac{2}{(s^2+1)(s^2+4)} = \frac{2}{3} \cdot \frac{1}{s^2+1} - \frac{2}{3} \cdot \frac{1}{s^2+4}$$

역 변환:

$$y(t) = \frac{2}{3}\sin t - \frac{1}{3}\sin 2t$$

### 5.4 연립 미분방정식

**예제**: 연립계

$$\begin{cases} x' = 2x - y, \quad x(0) = 1 \\ y' = x, \quad\quad\;\;\; y(0) = 0 \end{cases}$$

라플라스 변환 적용:

$$sX - 1 = 2X - Y \quad \Rightarrow \quad (s-2)X + Y = 1$$

$$sY = X \quad \Rightarrow \quad X = sY$$

대입하면:

$$s(s-2)Y + Y = 1 \quad \Rightarrow \quad Y(s) = \frac{1}{s^2 - 2s + 1} = \frac{1}{(s-1)^2}$$

$$X(s) = sY(s) = \frac{s}{(s-1)^2} = \frac{1}{s-1} + \frac{1}{(s-1)^2}$$

역 변환:

$$x(t) = e^t + te^t = (1 + t)e^t, \quad y(t) = te^t$$

```python
import sympy as sp

t, s = sp.symbols('t s')
Y = sp.Function('Y')

print("=== ODE 풀이: y'' + 3y' + 2y = 0, y(0)=1, y'(0)=0 ===\n")

# 방법 1: SymPy의 dsolve로 직접 풀기
y = sp.Function('y')
ode = sp.Eq(y(t).diff(t, 2) + 3*y(t).diff(t) + 2*y(t), 0)
sol = sp.dsolve(ode, y(t), ics={y(0): 1, y(t).diff(t).subs(t, 0): 0})
print(f"dsolve 결과: {sol}\n")

# 방법 2: 라플라스 변환을 단계별로 수행
print("--- 단계별 라플라스 변환 풀이 ---")

# Y(s) 구하기
Ys = (s + 3) / (s**2 + 3*s + 2)
print(f"Y(s) = {Ys}")

# 부분분수 분해
pf = sp.apart(Ys, s)
print(f"부분분수: {pf}")

# 역 라플라스 변환
yt = sp.inverse_laplace_transform(Ys, s, t)
print(f"y(t) = {yt}\n")

# 검증: 초기 조건과 ODE 만족 여부
print("--- 검증 ---")
print(f"y(0) = {yt.subs(t, 0)}")
print(f"y'(0) = {sp.diff(yt, t).subs(t, 0)}")
residual = sp.simplify(sp.diff(yt, t, 2) + 3*sp.diff(yt, t) + 2*yt)
print(f"y'' + 3y' + 2y = {residual}")

print("\n=== 비제차 ODE: y'' + y = sin(2t) ===\n")
Ys2 = 2 / ((s**2 + 1) * (s**2 + 4))
pf2 = sp.apart(Ys2, s)
print(f"부분분수: {pf2}")
yt2 = sp.inverse_laplace_transform(Ys2, s, t)
print(f"y(t) = {yt2}")
```

---

## 6. 전달함수와 시스템 해석

### 6.1 전달함수의 정의

선형 시불변(LTI) 시스템에서 **전달함수**는 입력과 출력의 라플라스 변환 비율이다:

$$H(s) = \frac{Y(s)}{X(s)}$$

여기서 초기 조건은 모두 0으로 가정한다. 전달함수는 시스템의 고유한 특성을 나타내며, 입력에 무관하다.

### 6.2 극점과 영점

- **영점(zeros)**: $H(s) = 0$이 되는 $s$ 값 (분자의 근)
- **극점(poles)**: $H(s) \to \infty$가 되는 $s$ 값 (분모의 근)

극점의 위치가 시스템의 동적 특성을 결정한다.

### 6.3 임펄스 응답과 계단 응답

- **임펄스 응답** $h(t) = \mathcal{L}^{-1}\{H(s)\}$: 입력이 $\delta(t)$일 때의 출력
- **계단 응답**: 입력이 $u(t)$일 때의 출력, $Y(s) = H(s)/s$

### 6.4 안정성 분석

시스템이 **BIBO 안정(Bounded-Input Bounded-Output stable)**하려면:

> 전달함수 $H(s)$의 **모든 극점이 복소 평면의 왼쪽 반평면**(즉, $\text{Re}(s) < 0$)에 위치해야 한다.

극점 위치에 따른 응답 특성:
- **왼쪽 반평면** ($\text{Re}(s) < 0$): 감쇠 -> 안정
- **허수축** ($\text{Re}(s) = 0$): 지속 진동 -> 한계 안정
- **오른쪽 반평면** ($\text{Re}(s) > 0$): 발산 -> 불안정

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 2차 시스템: H(s) = omega_n^2 / (s^2 + 2*zeta*omega_n*s + omega_n^2)
omega_n = 2.0  # 고유 진동수
zeta_values = [0.1, 0.3, 0.7, 1.0, 2.0]  # 감쇠비

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for zeta in zeta_values:
    # 전달함수 정의
    num = [omega_n**2]
    den = [1, 2*zeta*omega_n, omega_n**2]
    sys = signal.TransferFunction(num, den)

    # 계단 응답
    t_step, y_step = signal.step(sys)
    axes[0].plot(t_step, y_step, label=f'zeta={zeta}')

    # 극점 표시
    poles = np.roots(den)
    axes[1].plot(poles.real, poles.imag, 'x', markersize=10,
                 label=f'zeta={zeta}: {poles[0]:.2f}')

axes[0].set_xlabel('시간 t')
axes[0].set_ylabel('y(t)')
axes[0].set_title('계단 응답 (감쇠비에 따른 변화)')
axes[0].legend()
axes[0].grid(True)
axes[0].axhline(y=1, color='k', linestyle='--', alpha=0.3)

axes[1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
axes[1].set_xlabel('Re(s)')
axes[1].set_ylabel('Im(s)')
axes[1].set_title('극점 위치 (s-평면)')
axes[1].legend()
axes[1].grid(True)
axes[1].set_aspect('equal')

plt.tight_layout()
plt.savefig('transfer_function_analysis.png', dpi=150)
plt.show()
print("감쇠비가 클수록 극점이 실수축에 가까워지고, 응답이 빠르게 안정화됨")
```

---

## 7. 물리적 응용

### 7.1 RLC 회로 해석

직렬 RLC 회로에서 키르히호프 법칙을 적용하면:

$$L\frac{di}{dt} + Ri + \frac{1}{C}\int_0^t i(\tau) \, d\tau = V(t)$$

전하 $q$에 대해 ($i = dq/dt$):

$$L q'' + R q' + \frac{q}{C} = V(t)$$

라플라스 변환($q(0) = 0$, $q'(0) = 0$):

$$\left(Ls^2 + Rs + \frac{1}{C}\right) Q(s) = V(s)$$

**전달함수**:

$$H(s) = \frac{Q(s)}{V(s)} = \frac{1}{Ls^2 + Rs + \frac{1}{C}} = \frac{1/L}{s^2 + \frac{R}{L}s + \frac{1}{LC}}$$

자연 진동수 $\omega_0 = 1/\sqrt{LC}$, 감쇠비 $\zeta = R/(2\sqrt{L/C})$로 놓으면 표준 2차 시스템과 동일한 형태가 된다.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# RLC 회로 파라미터
R = 10       # Ohm (저항)
L = 0.1      # H (인덕턴스)
C = 1e-3     # F (커패시턴스)

omega_0 = 1 / np.sqrt(L * C)  # 고유 진동수
zeta = R / (2 * np.sqrt(L / C))  # 감쇠비
print(f"RLC 회로 파라미터:")
print(f"  고유 진동수 omega_0 = {omega_0:.2f} rad/s")
print(f"  감쇠비 zeta = {zeta:.4f}")
print(f"  상태: {'과감쇠' if zeta > 1 else '임계감쇠' if zeta == 1 else '부족감쇠'}")

# 전달함수: H(s) = (1/L) / (s^2 + (R/L)s + 1/(LC))
num = [1/L]
den = [1, R/L, 1/(L*C)]
sys_rlc = signal.TransferFunction(num, den)

# 단위 계단 전압 입력에 대한 응답 (스위치 ON)
t_sim = np.linspace(0, 0.1, 1000)
t_out, q_out = signal.step(sys_rlc, T=t_sim)

# 전류 i(t) = dq/dt
i_out = np.gradient(q_out, t_out)

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

axes[0].plot(t_out * 1000, q_out * 1e6, 'b-', linewidth=2)
axes[0].set_xlabel('시간 (ms)')
axes[0].set_ylabel('전하 q (uC)')
axes[0].set_title('RLC 직렬 회로 - 계단 응답')
axes[0].grid(True)

axes[1].plot(t_out * 1000, i_out * 1000, 'r-', linewidth=2)
axes[1].set_xlabel('시간 (ms)')
axes[1].set_ylabel('전류 i (mA)')
axes[1].set_title('전류 응답')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('rlc_circuit_response.png', dpi=150)
plt.show()
```

### 7.2 감쇠 진동 (질량-스프링-댐퍼 시스템)

질량 $m$, 스프링 상수 $k$, 감쇠 계수 $c$인 시스템:

$$m\ddot{x} + c\dot{x} + kx = F(t)$$

초기 변위 $x_0$에서 자유 진동($F = 0$)하는 경우:

$$ms^2 X - msx_0 + csX - cx_0 + kX = 0$$

$$X(s) = \frac{(ms + c)x_0}{ms^2 + cs + k} = \frac{x_0(s + c/m)}{s^2 + (c/m)s + k/m}$$

감쇠비 $\zeta = c/(2\sqrt{mk})$, 고유 진동수 $\omega_n = \sqrt{k/m}$, 감쇠 진동수 $\omega_d = \omega_n\sqrt{1-\zeta^2}$일 때:

$$x(t) = x_0 e^{-\zeta\omega_n t}\left(\cos\omega_d t + \frac{\zeta}{\sqrt{1-\zeta^2}}\sin\omega_d t\right)$$

```python
import numpy as np
import matplotlib.pyplot as plt

# 질량-스프링-댐퍼 파라미터
m = 1.0   # kg
k = 100.0  # N/m
x0 = 0.05  # m (초기 변위 5cm)

# 다양한 감쇠 조건
c_values = [0.5, 5.0, 20.0, 25.0]  # N*s/m

t = np.linspace(0, 3, 1000)
plt.figure(figsize=(10, 6))

for c in c_values:
    omega_n = np.sqrt(k / m)
    zeta = c / (2 * np.sqrt(m * k))

    if zeta < 1:  # 부족감쇠
        omega_d = omega_n * np.sqrt(1 - zeta**2)
        x = x0 * np.exp(-zeta * omega_n * t) * (
            np.cos(omega_d * t) +
            (zeta / np.sqrt(1 - zeta**2)) * np.sin(omega_d * t)
        )
        label = f'c={c} (zeta={zeta:.2f}, 부족감쇠)'
    elif zeta == 1:  # 임계감쇠
        x = x0 * (1 + omega_n * t) * np.exp(-omega_n * t)
        label = f'c={c} (zeta={zeta:.2f}, 임계감쇠)'
    else:  # 과감쇠
        r1 = -zeta * omega_n + omega_n * np.sqrt(zeta**2 - 1)
        r2 = -zeta * omega_n - omega_n * np.sqrt(zeta**2 - 1)
        A = x0 * r2 / (r2 - r1)
        B = -x0 * r1 / (r2 - r1)
        x = A * np.exp(r1 * t) + B * np.exp(r2 * t)
        label = f'c={c} (zeta={zeta:.2f}, 과감쇠)'

    plt.plot(t, x * 100, linewidth=2, label=label)

plt.xlabel('시간 (s)')
plt.ylabel('변위 (cm)')
plt.title('질량-스프링-댐퍼 시스템의 자유 진동')
plt.legend()
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.savefig('damped_oscillation.png', dpi=150)
plt.show()
```

### 7.3 열전도 문제

길이 $L$인 봉의 한쪽 끝에 갑자기 온도 $T_0$를 가할 때의 열전도 방정식:

$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$

시간에 대해 라플라스 변환을 적용하면 ($u(x, 0) = 0$):

$$sU(x, s) = \alpha \frac{d^2 U}{dx^2}$$

이는 $x$에 대한 상미분방정식이다:

$$U(x, s) = A e^{-x\sqrt{s/\alpha}} + B e^{x\sqrt{s/\alpha}}$$

경계 조건 $U(0, s) = T_0/s$, $U(\infty, s) = 0$을 적용하면:

$$U(x, s) = \frac{T_0}{s} e^{-x\sqrt{s/\alpha}}$$

역 변환하면:

$$u(x, t) = T_0 \, \text{erfc}\left(\frac{x}{2\sqrt{\alpha t}}\right)$$

여기서 $\text{erfc}$는 상보 오차 함수이다.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# 열전도 파라미터
T0 = 100.0        # 경계 온도 (deg C)
alpha = 1.14e-4   # 열확산계수 (m^2/s, 철)

# 시간별 온도 분포
x = np.linspace(0, 0.1, 200)  # 위치 (m)
times = [1, 10, 60, 300, 1800]  # 시간 (s)

plt.figure(figsize=(10, 6))
for t_val in times:
    u = T0 * erfc(x / (2 * np.sqrt(alpha * t_val)))
    plt.plot(x * 100, u, linewidth=2, label=f't = {t_val}s')

plt.xlabel('위치 x (cm)')
plt.ylabel('온도 u (deg C)')
plt.title('반무한 봉의 열전도 (라플라스 변환 풀이)')
plt.legend()
plt.grid(True)
plt.savefig('heat_conduction_laplace.png', dpi=150)
plt.show()
```

---

## 8. 수치적 역 라플라스 변환

### 8.1 필요성

많은 실제 문제에서 $F(s)$의 해석적 역 변환이 불가능하거나 매우 복잡하다. 이러한 경우 수치적 역 라플라스 변환 알고리즘이 필요하다.

### 8.2 스테페스트 알고리즘 (Stehfest Algorithm)

스테페스트 알고리즘은 실수축 위의 $F(s)$ 값만을 사용하여 $f(t)$를 근사하는 방법이다:

$$f(t) \approx \frac{\ln 2}{t} \sum_{k=1}^{N} V_k \, F\left(\frac{k \ln 2}{t}\right)$$

여기서 가중치 $V_k$는 이항계수를 이용하여 계산된다. $N$은 짝수(보통 $N = 10 \sim 18$)로 선택한다.

### 8.3 탈봇 알고리즘 (Talbot Method)

탈봇 방법은 브롬위치 적분의 경로를 변형한 것으로, 복소 평면에서의 포물선 경로를 따라 수치 적분한다. 스테페스트 방법보다 일반적으로 더 정확하다.

```python
import numpy as np
import matplotlib.pyplot as plt
from math import factorial

def stehfest_weights(N):
    """스테페스트 알고리즘의 가중치 계산"""
    V = np.zeros(N)
    for k in range(1, N + 1):
        s = 0
        for j in range(int((k + 1) / 2), min(k, N // 2) + 1):
            numer = j**(N // 2) * factorial(2 * j)
            denom = (factorial(N // 2 - j) *
                     factorial(j) *
                     factorial(j - 1) *
                     factorial(k - j) *
                     factorial(2 * j - k))
            s += numer / denom
        V[k - 1] = (-1)**(k + N // 2) * s
    return V

def numerical_inverse_laplace(F_func, t_values, N=12):
    """스테페스트 알고리즘을 이용한 수치적 역 라플라스 변환

    Parameters:
        F_func: F(s)를 계산하는 함수
        t_values: 역변환을 구할 시간값 배열
        N: 스테페스트 차수 (짝수, 기본값 12)
    """
    V = stehfest_weights(N)
    ln2 = np.log(2)
    result = np.zeros_like(t_values, dtype=float)

    for i, t in enumerate(t_values):
        if t <= 0:
            result[i] = 0
            continue
        s_vals = np.arange(1, N + 1) * ln2 / t
        F_vals = np.array([F_func(sv) for sv in s_vals])
        result[i] = (ln2 / t) * np.sum(V * F_vals)

    return result

# 검증: F(s) = 1/(s+1) -> f(t) = e^(-t)
F_exp = lambda sv: 1.0 / (sv + 1.0)
t_test = np.linspace(0.01, 5, 200)
f_numerical = numerical_inverse_laplace(F_exp, t_test)
f_exact = np.exp(-t_test)

plt.figure(figsize=(10, 6))
plt.plot(t_test, f_exact, 'b-', linewidth=2, label='해석적: exp(-t)')
plt.plot(t_test, f_numerical, 'r--', linewidth=2, label='스테페스트 (N=12)')
plt.xlabel('시간 t')
plt.ylabel('f(t)')
plt.title('수치적 역 라플라스 변환 검증')
plt.legend()
plt.grid(True)
plt.savefig('numerical_inverse_laplace.png', dpi=150)
plt.show()

# 오차 분석
max_error = np.max(np.abs(f_numerical - f_exact))
print(f"최대 오차: {max_error:.2e}")

# 더 복잡한 예: F(s) = 1/(s^2+1) -> f(t) = sin(t)
F_sin = lambda sv: 1.0 / (sv**2 + 1.0)
f_sin_numerical = numerical_inverse_laplace(F_sin, t_test)
f_sin_exact = np.sin(t_test)

print(f"\nsin(t) 역변환 최대 오차: {np.max(np.abs(f_sin_numerical - f_sin_exact)):.2e}")
```

---

## 연습 문제

### 기본

**문제 1.** 다음 함수의 라플라스 변환을 구하라.

(a) $f(t) = 3t^2 - 2e^{-t} + 5\cos(4t)$

(b) $f(t) = t^3 e^{2t}$

(c) $f(t) = e^{-3t}\sin(5t)$

**문제 2.** 다음 $F(s)$의 역 라플라스 변환을 구하라.

(a) $F(s) = \dfrac{5}{s^3}$

(b) $F(s) = \dfrac{2s + 1}{s^2 + 4s + 13}$

(c) $F(s) = \dfrac{3}{(s-1)(s+2)(s-3)}$

**문제 3.** 합성곱 정리를 이용하여 $\mathcal{L}^{-1}\left\{\dfrac{1}{s(s+1)}\right\}$를 구하라.

### 중급

**문제 4.** 라플라스 변환을 이용하여 다음 초기값 문제를 풀어라.

$$y'' - 4y' + 4y = e^{2t}, \quad y(0) = 0, \quad y'(0) = 1$$

**문제 5.** 제2이동 정리를 사용하여 다음의 라플라스 변환을 구하라.

$$f(t) = \begin{cases} 0, & 0 \leq t < 2 \\ t - 2, & t \geq 2 \end{cases}$$

**문제 6.** 다음 연립 미분방정식을 라플라스 변환으로 풀어라.

$$x' + y = e^t, \quad x + y' = 0, \quad x(0) = 1, \quad y(0) = 0$$

**문제 7.** 전달함수 $H(s) = \dfrac{s + 3}{s^2 + 4s + 8}$의 극점, 영점을 구하고 시스템의 안정성을 판별하라. 또한 임펄스 응답 $h(t)$를 구하라.

### 심화

**문제 8.** 라플라스 변환을 이용하여 적분방정식을 풀어라.

$$y(t) = 1 + \int_0^t y(\tau) \sin(t - \tau) \, d\tau$$

**문제 9.** 직렬 RLC 회로($R = 4\,\Omega$, $L = 1\,$H, $C = 1/5\,$F)에 $V(t) = 10u(t)$ (단위 계단 전압)을 인가할 때, 전류 $i(t)$를 라플라스 변환으로 구하라.

**문제 10.** 최종값 정리를 이용하여 다음 전달함수를 가진 시스템의 단위 계단 응답의 정상상태 값을 구하라.

$$H(s) = \frac{10(s + 2)}{s^2 + 5s + 6}$$

---

## 심화 학습

### 양측 라플라스 변환 (Bilateral Laplace Transform)

일반적인 라플라스 변환은 단측(unilateral)으로 $t \geq 0$에서만 정의되지만, **양측 라플라스 변환**은:

$$F(s) = \int_{-\infty}^{\infty} f(t) e^{-st} \, dt$$

양측 변환은 수렴 영역이 띠(strip) 형태가 되며, 신호 처리와 확률론에서 중요하다.

### Z-변환과의 관계

라플라스 변환의 이산 시간 대응물이 **Z-변환**이다:

$$X(z) = \sum_{n=0}^{\infty} x[n] z^{-n}$$

$z = e^{sT}$ ($T$는 샘플링 주기)로 놓으면 두 변환 사이의 관계가 성립한다. 디지털 신호 처리와 이산 제어 시스템에서 핵심적인 역할을 한다.

### 참고 자료

**교재**:
- **Boas, M. L.** *Mathematical Methods in the Physical Sciences*, 3rd ed., Ch. 8 (Sec. 10-12)
- **Arfken, Weber** *Mathematical Methods for Physicists*, Ch. 15

**보충 자료**:
- **Schiff, J. L.** *The Laplace Transform: Theory and Applications* - 이론과 응용을 균형있게 다룸
- **Dyke, P. P. G.** *An Introduction to Laplace Transforms and Fourier Series* - 물리수학적 접근

### 핵심 공식 요약

| 성질 | 시간 영역 | $s$-영역 |
|------|-----------|----------|
| 정의 | $f(t)$ | $F(s) = \int_0^\infty f(t)e^{-st}\,dt$ |
| 선형성 | $\alpha f + \beta g$ | $\alpha F + \beta G$ |
| 제1이동 | $e^{at}f(t)$ | $F(s-a)$ |
| 제2이동 | $f(t-a)u(t-a)$ | $e^{-as}F(s)$ |
| 미분 | $f'(t)$ | $sF(s) - f(0)$ |
| 적분 | $\int_0^t f(\tau)\,d\tau$ | $F(s)/s$ |
| 합성곱 | $(f*g)(t)$ | $F(s) \cdot G(s)$ |
| 초기값 | $f(0^+)$ | $\lim_{s\to\infty} sF(s)$ |
| 최종값 | $\lim_{t\to\infty} f(t)$ | $\lim_{s\to 0} sF(s)$ |

---

**이전**: [14. 복소해석](14_Complex_Analysis.md)
**다음**: [16. 그린 함수](16_Greens_Functions.md)
