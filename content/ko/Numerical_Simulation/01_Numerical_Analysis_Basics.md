# 수치해석 기초

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. IEEE 754 부동소수점(Floating-Point) 표현을 설명하고 반올림 오차(Round-off Error)와 절단 오차(Truncation Error)가 발생하는 원인을 파악할 수 있습니다.
2. 절대 오차(Absolute Error), 상대 오차(Relative Error), 기계 엡실론(Machine Epsilon)을 정의하고 수치 계산에 미치는 영향을 분석할 수 있습니다.
3. 유한 차분(Finite Difference) 근사를 사용한 수치 미분을 구현하고 정확도를 평가할 수 있습니다.
4. 수치 적분 방법(사다리꼴 공식, 심프슨 공식)을 적용하고 수렴 속도를 비교할 수 있습니다.
5. 수치 미분과 적분에서 스텝 크기와 수치 오차 사이의 트레이드오프를 분석할 수 있습니다.

---

## 개요

수치해석은 수학적 문제를 컴퓨터로 근사적으로 푸는 방법을 연구합니다. 시뮬레이션의 기초가 되는 부동소수점 표현, 오차 분석, 수치 미분과 적분을 학습합니다.

**이 레슨이 중요한 이유:** 모든 수치 시뮬레이션은 부동소수점 산술(floating-point arithmetic) 위에 구축되며, 모든 부동소수점 연산은 미세한 오차를 도입합니다. 이러한 오차는 누적되고, 상쇄되거나, 증폭되어 완전히 잘못된 결과를 만들어낼 수 있습니다. 컴퓨터가 수를 어떻게 표현하는지, 오차가 어디에서 발생하는지, 그리고 이를 어떻게 제어하는지 이해하는 것은 선택이 아닙니다 -- 신뢰할 수 있는 예측을 제공하는 시뮬레이션과 조용히 쓰레기 값을 생산하는 시뮬레이션의 차이를 결정짓기 때문입니다.

---

## 1. 부동소수점 표현

### 1.1 IEEE 754 표준

```python
import numpy as np
import struct

# 부동소수점 비트 표현 확인
def float_to_bits(f):
    """float64를 비트 문자열로 변환"""
    packed = struct.pack('>d', f)
    bits = ''.join(f'{b:08b}' for b in packed)
    return f"부호: {bits[0]} | 지수: {bits[1:12]} | 가수: {bits[12:]}"

print(float_to_bits(1.0))
print(float_to_bits(-1.0))
print(float_to_bits(0.1))

# 머신 엡실론
print(f"\nfloat64 머신 엡실론: {np.finfo(np.float64).eps}")
print(f"float32 머신 엡실론: {np.finfo(np.float32).eps}")
```

### 1.2 수치적 한계

```python
# 오버플로우와 언더플로우
print("float64 범위:")
print(f"  최소: {np.finfo(np.float64).min}")
print(f"  최대: {np.finfo(np.float64).max}")
print(f"  최소 양수: {np.finfo(np.float64).tiny}")

# 정밀도 손실 예시
a = 1e16
b = 1.0
print(f"\n1e16 + 1 - 1e16 = {(a + b) - a}")  # 0.0 (정밀도 손실)
print(f"1 + 1e16 - 1e16 = {b + (a - a)}")    # 1.0 (올바른 결과)
```

### 1.3 반올림 오차

```python
# 0.1을 정확히 표현할 수 없음
x = 0.1
print(f"0.1 실제 값: {x:.20f}")
print(f"0.1 + 0.2 = 0.3? {0.1 + 0.2 == 0.3}")  # False

# 비교 시 허용오차 사용
print(f"np.isclose: {np.isclose(0.1 + 0.2, 0.3)}")
```

---

## 2. 오차 분석

### 2.1 오차 유형

```python
def analyze_error(true_value, approx_value):
    """절대오차와 상대오차 계산"""
    abs_error = abs(true_value - approx_value)
    rel_error = abs_error / abs(true_value) if true_value != 0 else float('inf')
    return abs_error, rel_error

# 예시: π 근사
import math
approximations = [
    ("22/7", 22/7),
    ("355/113", 355/113),
    ("3.14159", 3.14159),
]

print("π 근사 오차 분석:")
for name, approx in approximations:
    abs_e, rel_e = analyze_error(math.pi, approx)
    print(f"  {name:10}: 절대오차={abs_e:.2e}, 상대오차={rel_e:.2e}")
```

### 2.2 수치 안정성

```python
# 불안정한 계산 예시: 큰 수에서 작은 차이
def unstable_subtract(x):
    """수치적으로 불안정한 뺄셈"""
    return (1 + x) - 1

def stable_subtract(x):
    """수치적으로 안정한 형태"""
    return x

x_values = [1e-15, 1e-16, 1e-17]
print("작은 수 뺄셈 비교:")
for x in x_values:
    print(f"  x={x}: 불안정={unstable_subtract(x):.2e}, 안정={stable_subtract(x):.2e}")
```

### 2.3 조건수 (Condition Number)

```python
# 행렬의 조건수
def analyze_condition_number():
    # 잘 조건화된 행렬
    A_good = np.array([[1, 0], [0, 1]])

    # 나쁘게 조건화된 행렬
    A_bad = np.array([[1, 1], [1, 1.0001]])

    print("조건수 분석:")
    print(f"  단위 행렬: {np.linalg.cond(A_good):.2f}")
    print(f"  거의 특이 행렬: {np.linalg.cond(A_bad):.2f}")

analyze_condition_number()
```

---

## 3. 수치 미분

미분은 모든 PDE 솔버의 기초입니다. 이산 격자점에서의 함수값을 사용하여 도함수를 근사합니다. 세 가지 주요 공식은 정확도와 사용하는 점에서 차이가 있습니다:

- **전진 차분(Forward difference)**: $f'(x) \approx \frac{f(x+h) - f(x)}{h} + O(h)$ -- 1차 정확도
- **후진 차분(Backward difference)**: $f'(x) \approx \frac{f(x) - f(x-h)}{h} + O(h)$ -- 1차 정확도
- **중심 차분(Central difference)**: $f'(x) \approx \frac{f(x+h) - f(x-h)}{2h} + O(h^2)$ -- 2차 정확도

중심 차분이 더 정확한 이유는 테일러 전개(Taylor expansion)에서 1차 오차항이 대칭성에 의해 상쇄되기 때문입니다.

### 3.1 유한차분법

```python
def numerical_derivatives(f, x, h=1e-5):
    """다양한 유한차분 공식"""
    # 전진차분: x와 x+h의 함수값만 사용 (한쪽만, O(h))
    forward = (f(x + h) - f(x)) / h

    # 후진차분: x와 x-h의 함수값만 사용 (한쪽만, O(h))
    backward = (f(x) - f(x - h)) / h

    # 중심차분: x를 중심으로 대칭, 오차항이 상쇄 → O(h²)
    central = (f(x + h) - f(x - h)) / (2 * h)

    return forward, backward, central

# 테스트: f(x) = sin(x), f'(x) = cos(x)
x = np.pi / 4
true_deriv = np.cos(x)

forward, backward, central = numerical_derivatives(np.sin, x)

print(f"x = π/4에서 sin(x)의 도함수:")
print(f"  참값: {true_deriv:.10f}")
print(f"  전진차분: {forward:.10f}, 오차: {abs(forward - true_deriv):.2e}")
print(f"  후진차분: {backward:.10f}, 오차: {abs(backward - true_deriv):.2e}")
print(f"  중심차분: {central:.10f}, 오차: {abs(central - true_deriv):.2e}")
```

### 3.2 고차 미분

```python
def second_derivative(f, x, h=1e-5):
    """2차 도함수 (중심차분)"""
    return (f(x + h) - 2*f(x) + f(x - h)) / h**2

# f(x) = sin(x), f''(x) = -sin(x)
x = np.pi / 4
true_second = -np.sin(x)
approx_second = second_derivative(np.sin, x)

print(f"\n2차 도함수:")
print(f"  참값: {true_second:.10f}")
print(f"  근사값: {approx_second:.10f}")
print(f"  오차: {abs(approx_second - true_second):.2e}")
```

### 3.3 스텝 크기의 영향

절단 오차(truncation error, 작은 $h$에서 감소)와 반올림 오차(rounding error, 작은 $h$에서 거의 같은 수를 빼므로 증가) 사이에 근본적인 긴장이 있습니다. 최적 스텝 크기는 이 둘의 균형점입니다: 중심 차분의 경우 $h_{\text{opt}} \approx \varepsilon_{\text{mach}}^{1/3} \approx 6 \times 10^{-6}$ (float64 기준).

```python
def analyze_step_size():
    """스텝 크기에 따른 오차 분석"""
    f = np.sin
    x = 1.0
    true_deriv = np.cos(x)

    h_values = np.logspace(-1, -15, 15)
    errors = []

    for h in h_values:
        central = (f(x + h) - f(x - h)) / (2 * h)
        errors.append(abs(central - true_deriv))

    return h_values, errors

h_values, errors = analyze_step_size()

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.loglog(h_values, errors, 'bo-')
plt.xlabel('Step size h')
plt.ylabel('Error')
plt.title('중심차분 오차 vs 스텝 크기')
plt.grid(True)
plt.axvline(x=1e-8, color='r', linestyle='--', label='최적 근처')
plt.legend()
plt.show()
# 너무 작은 h: 반올림 오차 증가
# 너무 큰 h: 절단 오차 증가
```

---

## 4. 수치 적분

수치 적분(구적법, quadrature)은 적분을 해석적으로 계산할 수 없을 때 필요합니다. 기본 아이디어: 각 부분 구간에서 피적분함수를 다항식으로 근사한 후 다항식을 정확하게 적분합니다. 고차 다항식일수록 더 빠른 수렴을 제공합니다: 사다리꼴 공식은 선형 보간을 사용하고($O(h^2)$), 심슨 공식은 2차 보간을 사용하며($O(h^4)$), 가우스 구적법(Gauss quadrature)은 더 높은 차수를 달성합니다.

### 4.1 사다리꼴 공식

사다리꼴 공식은 인접한 점들을 직선으로 연결하여 곡선 아래 면적을 근사합니다. 각 부분 구간 $[x_i, x_{i+1}]$에서 면적은 사다리꼴처럼 $h \cdot (f_i + f_{i+1})/2$입니다.

```python
def trapezoidal(f, a, b, n):
    """사다리꼴 공식으로 적분"""
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    # 사다리꼴 공식
    integral = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    return integral

# 테스트: ∫₀^π sin(x) dx = 2
result = trapezoidal(np.sin, 0, np.pi, 100)
print(f"∫₀^π sin(x) dx:")
print(f"  참값: 2.0")
print(f"  사다리꼴 (n=100): {result:.10f}")
print(f"  오차: {abs(result - 2.0):.2e}")
```

### 4.2 심슨 공식

심슨 공식은 연속 세 점을 통과하는 2차 다항식을 맞추어 $O(h^4)$ 수렴을 달성합니다 -- 같은 수의 점으로 사다리꼴 공식보다 두 차수나 더 좋습니다. 이 극적인 향상은 2차 다항식이 3차 다항식을 정확히 적분하기 때문입니다 (오차 분석에서의 행운의 상쇄).

```python
def simpson(f, a, b, n):
    """심슨 1/3 공식 (n은 짝수여야 함)"""
    if n % 2 != 0:
        n += 1

    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    # 심슨 공식: (h/3) * [y₀ + 4y₁ + 2y₂ + 4y₃ + ... + yₙ]
    integral = h/3 * (y[0] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-1:2]) + y[-1])
    return integral

result_trap = trapezoidal(np.sin, 0, np.pi, 10)
result_simp = simpson(np.sin, 0, np.pi, 10)

print(f"\nn=10에서 비교:")
print(f"  사다리꼴: {result_trap:.10f}, 오차: {abs(result_trap - 2.0):.2e}")
print(f"  심슨: {result_simp:.10f}, 오차: {abs(result_simp - 2.0):.2e}")
```

### 4.3 SciPy 적분

```python
from scipy import integrate

# 1차원 적분
result, error = integrate.quad(np.sin, 0, np.pi)
print(f"\nscipy.integrate.quad:")
print(f"  결과: {result:.15f}")
print(f"  추정 오차: {error:.2e}")

# 2차원 적분
def f_2d(y, x):
    return x * y

result_2d, error_2d = integrate.dblquad(f_2d, 0, 1, 0, 1)
print(f"\n∫∫ xy dxdy (0~1):")
print(f"  결과: {result_2d:.10f}")  # 0.25
```

### 4.4 수렴 분석

```python
def convergence_analysis():
    """적분 수렴 분석"""
    true_value = 2.0  # ∫₀^π sin(x) dx
    n_values = [4, 8, 16, 32, 64, 128, 256]

    trap_errors = []
    simp_errors = []

    for n in n_values:
        trap_errors.append(abs(trapezoidal(np.sin, 0, np.pi, n) - true_value))
        simp_errors.append(abs(simpson(np.sin, 0, np.pi, n) - true_value))

    # 수렴 차수 추정
    print("수렴 분석:")
    print(f"{'n':>6} {'사다리꼴':>12} {'심슨':>12}")
    for i, n in enumerate(n_values):
        print(f"{n:>6} {trap_errors[i]:>12.2e} {simp_errors[i]:>12.2e}")

    # 사다리꼴: O(h²), 심슨: O(h⁴)
    return n_values, trap_errors, simp_errors

convergence_analysis()
```

---

## 5. 연습 문제

### 문제 1: 수치 미분
함수 f(x) = e^(-x²)의 도함수를 x=0.5에서 다양한 스텝 크기로 계산하고 오차를 분석하세요.

```python
def exercise_1():
    f = lambda x: np.exp(-x**2)
    f_prime = lambda x: -2*x * np.exp(-x**2)  # 해석적 도함수

    x = 0.5
    true_value = f_prime(x)

    # 풀이
    h_values = np.logspace(-1, -12, 12)
    for h in h_values:
        approx = (f(x + h) - f(x - h)) / (2 * h)
        print(f"h={h:.0e}: 오차={abs(approx - true_value):.2e}")

exercise_1()
```

### 문제 2: 수치 적분
∫₀^1 e^(-x²) dx를 사다리꼴과 심슨 공식으로 계산하세요.

```python
def exercise_2():
    f = lambda x: np.exp(-x**2)

    # scipy 참값
    true_val, _ = integrate.quad(f, 0, 1)

    # 풀이
    for n in [10, 50, 100]:
        trap = trapezoidal(f, 0, 1, n)
        simp = simpson(f, 0, 1, n)
        print(f"n={n}: 사다리꼴={trap:.8f}, 심슨={simp:.8f}")

    print(f"참값: {true_val:.8f}")

exercise_2()
```

---

## 요약

| 개념 | 핵심 내용 |
|------|----------|
| 부동소수점 | IEEE 754, 머신 엡실론, 정밀도 한계 |
| 오차 유형 | 절단 오차, 반올림 오차, 조건수 |
| 수치 미분 | 전진/후진/중심차분, 스텝 크기 선택 |
| 수치 적분 | 사다리꼴(O(h²)), 심슨(O(h⁴)) |

## 연습 문제

### 연습 문제 1: 머신 엡실론(Machine Epsilon)과 정밀도 손실

`float64`와 `float32`에서 머신 엡실론(ε_mach)이 무엇을 나타내는지 설명하세요. 그런 다음 파국적 소거(catastrophic cancellation) 문제를 시연하세요: `a = 1e15`, `b = 1.0`에 대해 `(a + b) - a`를 계산하고, 결과가 왜 잘못되었는지 설명한 후, 문제를 피하는 방식으로 식을 재작성하세요.

<details>
<summary>정답 보기</summary>

머신 엡실론은 부동소수점 산술에서 `1.0 + ε ≠ 1.0`이 되는 가장 작은 양수 ε입니다. 단일 연산의 상대적 반올림 오차의 상한을 나타냅니다.

```python
import numpy as np

# 머신 엡실론 값
print(f"float64 머신 엡실론: {np.finfo(np.float64).eps:.2e}")  # ~2.22e-16
print(f"float32 머신 엡실론: {np.finfo(np.float32).eps:.2e}")  # ~1.19e-07

# 파국적 소거
a = 1e15
b = 1.0

# 잘못된 방법: a와 b는 15자릿수 차이가 남
result_bad = (a + b) - a
print(f"\n(a + b) - a = {result_bad}")   # 0.0  -- b의 정밀도가 손실됨

# 올바른 방법: 덧셈 전에 뺄셈 수행
result_good = b + (a - a)
print(f"b + (a - a) = {result_good}")    # 1.0  -- 수학적으로 동일하지만 수치적으로 안정

# a >> b일 때, stable_form = b를 직접 사용
# 핵심: 큰 수에 작은 수를 더하는 것을 피하도록 연산 순서를 재배열
```

근본 원인: `a`가 float64로 저장될 때 약 15~16개의 유효 십진수만 가집니다. `a = 1e15`에 `b = 1.0`을 더하려면 소수점 위치를 정렬해야 하므로, `b`의 비트가 표현 가능한 범위 아래로 밀려나 `b`가 완전히 소실됩니다.

</details>

### 연습 문제 2: 전진 차분 vs. 중심 차분 오차 스케일링

`x = 0.7`에서 `f(x) = cos(x)`의 수치 도함수를 전진 차분(`O(h)`)과 중심 차분(`O(h²)`) 공식을 사용하여 스텝 크기 `h = 10⁻¹, 10⁻³, 10⁻⁵, 10⁻⁷`에서 계산하세요. 각 방법에 대해 해석값 `f'(x) = -sin(x)`와의 오차를 계산하고, h를 100배 줄일 때 오차 비율이 예상 수렴 차수와 일치하는지 확인하세요.

<details>
<summary>정답 보기</summary>

```python
import numpy as np

x = 0.7
true_val = -np.sin(x)
h_values = [1e-1, 1e-3, 1e-5, 1e-7]

print(f"참 도함수: {true_val:.10f}\n")
print(f"{'h':>8}  {'전진 오차':>14}  {'중심 오차':>14}")
print("-" * 42)

prev_fwd, prev_cen = None, None
for h in h_values:
    fwd  = (np.cos(x + h) - np.cos(x)) / h
    cen  = (np.cos(x + h) - np.cos(x - h)) / (2 * h)
    e_fwd = abs(fwd - true_val)
    e_cen = abs(cen - true_val)
    print(f"{h:8.0e}  {e_fwd:14.2e}  {e_cen:14.2e}", end="")
    if prev_fwd is not None:
        # 비율은 ~100 (O(h))이고 h가 100배 줄면 ~10000 (O(h²))이어야 함
        print(f"  [전진 비율: {prev_fwd/e_fwd:.0f}, 중심 비율: {prev_cen/e_cen:.0f}]", end="")
    print()
    prev_fwd, prev_cen = e_fwd, e_cen
```

예상 출력 패턴:
- 전진 차분 오차 비율 ≈ 100 (오차가 h에 선형적으로 감소 → O(h))
- 중심 차분 오차 비율 ≈ 10000 (오차가 h의 제곱으로 감소 → O(h²))

참고: 매우 작은 h (전진의 경우 ≈ 1e-7, 중심의 경우 ≈ 1e-5)에서 반올림 오차가 지배하기 시작하여 비율이 깨집니다.

</details>

### 연습 문제 3: 복합 심슨 공식(Composite Simpson's Rule) 수렴

복합 심슨 공식을 구현하고 n = 4, 8, 16, 32, 64 분할에서 `∫₀^1 x³ e^x dx`를 계산하여 O(h⁴) 수렴 차수를 수치적으로 검증하세요. 정확한 값은 `e - 2 ≈ 0.71828...`입니다. 연속적인 오차 비율로부터 추정 수렴 차수를 계산하세요.

<details>
<summary>정답 보기</summary>

```python
import numpy as np
from scipy.integrate import quad

def simpson(f, a, b, n):
    """복합 심슨 1/3 공식 (n은 짝수여야 함)."""
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return h / 3 * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]) + y[-1])

f = lambda x: x**3 * np.exp(x)
exact, _ = quad(f, 0, 1)   # ≈ e - 2

print(f"정확한 값: {exact:.10f}")
print(f"\n{'n':>6}  {'근사값':>14}  {'오차':>12}  {'수렴 차수':>10}")
print("-" * 48)

prev_error = None
for n in [4, 8, 16, 32, 64]:
    approx = simpson(f, 0, 1, n)
    error  = abs(approx - exact)
    if prev_error is not None:
        order = np.log2(prev_error / error)
        print(f"{n:>6}  {approx:14.10f}  {error:12.2e}  {order:10.2f}")
    else:
        print(f"{n:>6}  {approx:14.10f}  {error:12.2e}  {'—':>10}")
    prev_error = error
```

출력되는 수렴 차수는 **4.0**에 가까워야 하며, O(h⁴) = O((1/n)⁴)를 확인합니다. n을 2배로 늘리면 오차가 약 16배 줄어듭니다.

</details>

### 연습 문제 4: 조건수(Condition Number)와 선형 시스템 민감도

조건수가 10⁵보다 큰 거의 특이한(nearly-singular) 2×2 행렬을 만들고 `Ax = b`를 두 번 풀어보세요: 한 번은 정확한 `b`로, 한 번은 크기 ≈ 10⁻⁶의 작은 벡터로 섭동된(perturbed) `b`로. 해의 상대 오차가 `b`의 상대 오차보다 훨씬 클 수 있음을 보이고, 조건수 한계를 사용하여 이 증폭을 정량화하세요.

<details>
<summary>정답 보기</summary>

```python
import numpy as np

# 힐베르트 행렬 (악명 높은 불량 조건 행렬)
n = 5
A = np.array([[1 / (i + j + 1) for j in range(n)] for i in range(n)])
cond = np.linalg.cond(A)
print(f"5×5 힐베르트 행렬의 조건수: {cond:.2e}")

# 정확한 우변: x_true = [1, 1, 1, 1, 1]
x_true = np.ones(n)
b_exact = A @ x_true

# 섭동된 우변
rng = np.random.default_rng(0)
delta_b = rng.standard_normal(n)
delta_b *= 1e-6 / np.linalg.norm(delta_b)   # ||δb|| ≈ 1e-6

b_perturbed = b_exact + delta_b

# 두 시스템 풀기
x_exact     = np.linalg.solve(A, b_exact)
x_perturbed = np.linalg.solve(A, b_perturbed)

rel_b_error = np.linalg.norm(delta_b) / np.linalg.norm(b_exact)
rel_x_error = np.linalg.norm(x_perturbed - x_exact) / np.linalg.norm(x_exact)

print(f"\nb의 상대 오차:  {rel_b_error:.2e}")
print(f"x의 상대 오차:  {rel_x_error:.2e}")
print(f"증폭 인수: {rel_x_error / rel_b_error:.1f}")
print(f"조건수:    {cond:.2e}")
print(f"\n이론: ||δx||/||x|| ≤ κ(A) * ||δb||/||b||")
print(f"한계:  {cond * rel_b_error:.2e}  (실제: {rel_x_error:.2e})")
```

`x`의 상대 오차는 `b`의 상대 오차보다 최대 `κ(A)`배 커질 수 있습니다. 5×5 힐베르트 행렬의 경우 κ ≈ 5×10⁵이므로, `b`에서 10⁻⁶ 크기의 섭동이 `x`에서 10⁻¹ 크기의 오차를 만들 수 있습니다.

</details>

### 연습 문제 5: 중심 차분의 최적 스텝 크기

`x = 1.0`에서 `f(x) = sin(x)`에 대해 스텝 크기 범위 `h = 10⁻¹`에서 `h = 10⁻¹⁵`까지 중심 차분을 사용하여 로그-로그 스케일로 오차 vs. h를 그리세요. 절단 오차와 반올림 오차가 균형을 이루는 근사 최적 스텝 크기를 찾고, 이론적 최적값 `h_opt ≈ (ε_mach)^(1/3)`을 유도하세요.

<details>
<summary>정답 보기</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

f      = np.sin
x      = 1.0
f_true = np.cos(x)   # 정확한 도함수

h_values = np.logspace(-1, -15, 150)
errors   = [abs((f(x + h) - f(x - h)) / (2 * h) - f_true) for h in h_values]

# 이론적 분석:
# 절단 오차  ≈ h² |f'''(x)| / 6   → h²에 비례하여 증가
# 반올림 오차 ≈ ε_mach |f(x)| / h  → 1/h에 비례하여 증가
# 최적 균형:  h_opt ≈ (ε_mach)^(1/3) ≈ 6e-6

eps   = np.finfo(float).eps
h_opt = eps ** (1/3)
print(f"머신 엡실론:  {eps:.2e}")
print(f"이론적 h_opt ≈ ε^(1/3) = {h_opt:.2e}")

plt.figure(figsize=(9, 5))
plt.loglog(h_values, errors, 'b-', linewidth=1.5, label='실제 오차')
plt.axvline(h_opt, color='r', linestyle='--', label=f'이론적 h_opt ≈ {h_opt:.0e}')
plt.xlabel('스텝 크기 h')
plt.ylabel('절대 오차')
plt.title('중심 차분 오차 vs. 스텝 크기')
plt.legend()
plt.grid(True, which='both', alpha=0.4)
plt.show()

# 관측된 최솟값
min_idx = np.argmin(errors)
print(f"관측된 최적 h ≈ {h_values[min_idx]:.2e}  (최소 오차: {errors[min_idx]:.2e})")
```

오차 곡선은 두 개의 구간을 보입니다:
- **최솟값 왼쪽** (큰 h): 절단 오차가 지배, 로그-로그 플롯에서 기울기 ≈ +2.
- **최솟값 오른쪽** (작은 h): 반올림 오차가 지배, 기울기 ≈ -1.

이론적 최적 스텝은 `h_opt = (ε_mach)^{1/3} ≈ 6×10⁻⁶`이며, 최소 오차는 `ε_mach^{2/3} ≈ 4×10⁻¹¹` 근처입니다.

</details>
