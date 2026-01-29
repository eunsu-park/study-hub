# 수치해석 기초

## 개요

수치해석은 수학적 문제를 컴퓨터로 근사적으로 푸는 방법을 연구합니다. 시뮬레이션의 기초가 되는 부동소수점 표현, 오차 분석, 수치 미분과 적분을 학습합니다.

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

### 3.1 유한차분법

```python
def numerical_derivatives(f, x, h=1e-5):
    """다양한 유한차분 공식"""
    # 전진차분 (forward difference)
    forward = (f(x + h) - f(x)) / h

    # 후진차분 (backward difference)
    backward = (f(x) - f(x - h)) / h

    # 중심차분 (central difference) - 더 정확함
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

### 4.1 사다리꼴 공식

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
