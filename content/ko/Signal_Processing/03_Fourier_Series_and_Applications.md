# 푸리에 급수와 응용

**이전**: [02. LTI 시스템과 합성곱](./02_LTI_Systems_and_Convolution.md) | **다음**: [04. 연속 푸리에 변환](./04_Continuous_Fourier_Transform.md)

---

레슨 02에서 LTI 시스템이 임펄스 응답에 의해 완전히 특성화되며, 합성곱을 통해 임의의 입력에 대한 출력을 계산할 수 있음을 살펴보았습니다. 하지만 시간 영역에서의 합성곱은 번거로울 수 있습니다. 훨씬 더 우아한 접근 방법이 있습니다. 신호를 복소 지수함수로 분해하는 것인데, 복소 지수함수는 LTI 시스템의 **고유함수(eigenfunction)**입니다. 각 지수함수에 대한 응답은 단순한 스케일링이며, 전체 출력은 이 스케일된 지수함수들의 합입니다.

**주기 신호(periodic signal)**의 경우, 이 분해를 **푸리에 급수(Fourier series)**라고 합니다. 이 레슨에서는 수학적 기초로부터 푸리에 급수를 유도하고, 수렴 특성을 탐구하며, 주기 현상 분석에서의 활용을 보여줍니다.

**난이도**: ⭐⭐⭐

**학습 목표**:
- 푸리에 급수의 삼각함수 형태와 복소 지수 형태 유도
- 표준 파형에 대한 푸리에 계수 계산
- 수렴을 위한 디리클레 조건(Dirichlet conditions) 진술 및 적용
- 깁스 현상(Gibbs phenomenon)과 실용적 시사점 설명
- 파르세발 정리(Parseval's theorem)를 활용한 스펙트럼 계수로부터 신호 전력 계산
- 선 스펙트럼(line spectra, 진폭 및 위상 스펙트럼) 해석
- Python을 활용한 푸리에 급수 근사 계산 및 수렴 시각화

---

## 목차

1. [주기 신호 복습](#1-주기-신호-복습)
2. [삼각함수 푸리에 급수](#2-삼각함수-푸리에-급수)
3. [복소 지수 푸리에 급수](#3-복소-지수-푸리에-급수)
4. [푸리에 계수 계산](#4-푸리에-계수-계산)
5. [푸리에 급수의 수렴](#5-푸리에-급수의-수렴)
6. [깁스 현상](#6-깁스-현상)
7. [파르세발 정리](#7-파르세발-정리)
8. [선 스펙트럼](#8-선-스펙트럼)
9. [응용: 표준 파형 분해](#9-응용-표준-파형-분해)
10. [이산시간 푸리에 급수](#10-이산시간-푸리에-급수)
11. [Python 예제](#11-python-예제)
12. [요약](#12-요약)
13. [연습 문제](#13-연습-문제)
14. [참고 문헌](#14-참고-문헌)

---

## 1. 주기 신호 복습

### 1.1 정의

연속시간 신호 $x(t)$가 주기 $T$를 가진 **주기 신호**인 경우:

$$x(t + T) = x(t) \quad \text{모든 } t \text{에 대해}$$

이를 만족하는 가장 작은 양수 $T$를 **기본 주기(fundamental period)** $T_0$라고 합니다. **기본 주파수(fundamental frequency)**는:

$$f_0 = \frac{1}{T_0} \quad \text{(Hz)}, \qquad \omega_0 = \frac{2\pi}{T_0} \quad \text{(rad/s)}$$

### 1.2 고조파

기본 주파수 $\omega_0$를 가진 주기 신호의 **$n$번째 고조파(harmonic)**는 주파수 $n\omega_0$를 가집니다. 기본파는 첫 번째 고조파($n = 1$)입니다. $2\omega_0$에서의 신호는 두 번째 고조파, 이런 식으로 이어집니다.

### 1.3 주기 신호의 합

$x_1(t)$의 주기가 $T_1$이고 $x_2(t)$의 주기가 $T_2$이면, 그 합 $x_1(t) + x_2(t)$는 $T_1/T_2$가 **유리수(rational number)**인 경우에만 주기적입니다. 합의 주기는 $T_1$과 $T_2$의 **최소공배수(least common multiple)**입니다.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Periodic signals and harmonics ---
t = np.linspace(0, 4, 2000)

# Fundamental and harmonics
f0 = 1.0  # 1 Hz fundamental
fig, axes = plt.subplots(4, 1, figsize=(12, 10))

for n, ax in zip([1, 2, 3, 5], axes):
    signal = np.cos(2 * np.pi * n * f0 * t)
    ax.plot(t, signal, linewidth=1.5)
    ax.set_title(f'Harmonic n={n}: $\\cos(2\\pi \\cdot {n} \\cdot {f0} \\cdot t)$, '
                 f'frequency = {n * f0} Hz, period = {1/(n*f0):.3f} s')
    ax.set_xlabel('t (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 4])

plt.tight_layout()
plt.savefig('harmonics.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 2. 삼각함수 푸리에 급수

### 2.1 표현

주기 $T_0$를 가진 주기 신호 $x(t)$를 표현하는 **삼각함수 푸리에 급수(trigonometric Fourier series)**:

$$x(t) = a_0 + \sum_{n=1}^{\infty} \left[ a_n \cos(n\omega_0 t) + b_n \sin(n\omega_0 t) \right]$$

여기서 $\omega_0 = 2\pi/T_0$이고 **푸리에 계수(Fourier coefficients)**는:

$$a_0 = \frac{1}{T_0} \int_{T_0} x(t) \, dt \quad \text{(DC 성분 / 평균값)}$$

$$a_n = \frac{2}{T_0} \int_{T_0} x(t) \cos(n\omega_0 t) \, dt, \quad n = 1, 2, 3, \ldots$$

$$b_n = \frac{2}{T_0} \int_{T_0} x(t) \sin(n\omega_0 t) \, dt, \quad n = 1, 2, 3, \ldots$$

$\int_{T_0}$은 임의의 완전한 한 주기에 대한 적분을 의미합니다.

### 2.2 직교성으로부터의 유도

계수를 구하는 핵심은 한 주기 $[0, T_0]$에 걸친 삼각함수의 **직교성(orthogonality)**입니다:

$$\int_0^{T_0} \cos(m\omega_0 t) \cos(n\omega_0 t) \, dt = \begin{cases} T_0 & m = n = 0 \\ T_0/2 & m = n \neq 0 \\ 0 & m \neq n \end{cases}$$

$$\int_0^{T_0} \sin(m\omega_0 t) \sin(n\omega_0 t) \, dt = \begin{cases} T_0/2 & m = n \neq 0 \\ 0 & m \neq n \end{cases}$$

$$\int_0^{T_0} \cos(m\omega_0 t) \sin(n\omega_0 t) \, dt = 0 \quad \text{모든 } m, n \text{에 대해}$$

$a_n$을 구하려면: 푸리에 급수의 양변에 $\cos(n\omega_0 t)$를 곱하고 한 주기에 걸쳐 적분합니다. 직교성 덕분에 $n$과 일치하는 항을 제외한 모든 항이 사라지고, 계수 공식이 도출됩니다.

### 2.3 콤팩트 삼각함수 형태

각 고조파를 하나의 정현파로 결합할 수 있습니다:

$$x(t) = C_0 + \sum_{n=1}^{\infty} C_n \cos(n\omega_0 t + \phi_n)$$

여기서:

$$C_0 = a_0, \quad C_n = \sqrt{a_n^2 + b_n^2}, \quad \phi_n = -\arctan\left(\frac{b_n}{a_n}\right)$$

### 2.4 대칭성을 이용한 계산 단순화

신호의 대칭성을 이용하면 계수 계산을 단순화할 수 있습니다:

| 신호 특성 | 결과 |
|-----------|------|
| **우함수(Even)**: $x(t) = x(-t)$ | 모든 $n$에 대해 $b_n = 0$ (코사인 항만 존재) |
| **기함수(Odd)**: $x(t) = -x(-t)$ | 모든 $n$에 대해 $a_n = 0$ (사인 항만 존재, $a_0 = 0$) |
| **반파 대칭(Half-wave symmetry)**: $x(t + T_0/2) = -x(t)$ | 짝수 $n$에 대해 $a_n = b_n = 0$ (홀수 고조파만 존재) |

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Orthogonality verification ---
T0 = 2 * np.pi  # period
omega0 = 2 * np.pi / T0  # = 1
t = np.linspace(0, T0, 10000)
dt = t[1] - t[0]

print("=== Orthogonality of Trigonometric Functions ===\n")

# cos(m*t) * cos(n*t) integrals
print("cos(m*t) * cos(n*t) over [0, 2*pi]:")
for m in range(4):
    for n in range(4):
        integral = np.trapz(np.cos(m * t) * np.cos(n * t), t)
        if abs(integral) > 1e-8:
            print(f"  m={m}, n={n}: {integral:.4f} (expected {T0 if m==n==0 else T0/2:.4f})")

print("\ncos(m*t) * sin(n*t) over [0, 2*pi]:")
all_zero = True
for m in range(4):
    for n in range(1, 4):
        integral = np.trapz(np.cos(m * t) * np.sin(n * t), t)
        if abs(integral) > 1e-8:
            all_zero = False
            print(f"  m={m}, n={n}: {integral:.4f}")
if all_zero:
    print("  All integrals are zero (as expected)")
```

---

## 3. 복소 지수 푸리에 급수

### 3.1 표현

**복소 지수 푸리에 급수(complex exponential Fourier series)**(흔히 **지수 푸리에 급수**라고도 함):

$$x(t) = \sum_{n=-\infty}^{\infty} c_n \, e^{jn\omega_0 t}$$

여기서 **복소 푸리에 계수(complex Fourier coefficients)**는:

$$c_n = \frac{1}{T_0} \int_{T_0} x(t) \, e^{-jn\omega_0 t} \, dt, \quad n = 0, \pm 1, \pm 2, \ldots$$

### 3.2 삼각함수 계수와의 관계

$$c_0 = a_0$$

$$c_n = \frac{a_n - jb_n}{2}, \quad c_{-n} = \frac{a_n + jb_n}{2} = c_n^* \quad (n > 0)$$

역관계:

$$a_n = 2\text{Re}(c_n) = c_n + c_{-n}$$

$$b_n = -2\text{Im}(c_n) = j(c_n - c_{-n})$$

**실수 신호**의 경우: $c_{-n} = c_n^*$ (켤레 대칭, conjugate symmetry).

### 3.3 복소 형태를 사용하는 이유

복소 지수 형태는 신호처리에서 다음과 같은 이유로 선호됩니다:

1. **간결한 표기**: $a_n$과 $b_n$을 따로 쓰는 대신 단일 합산
2. **고유함수 특성**: $e^{jn\omega_0 t}$는 LTI 시스템의 고유함수
3. **푸리에 변환으로 자연스러운 확장**: 변환은 연속 주파수 확장
4. **주파수 성분의 직접 가시화**: 주파수 $n\omega_0$에서의 $c_n$이 크기와 위상 모두 제공

### 3.4 유도

오일러 공식(Euler's formula)으로 시작:

$$\cos(n\omega_0 t) = \frac{e^{jn\omega_0 t} + e^{-jn\omega_0 t}}{2}, \quad \sin(n\omega_0 t) = \frac{e^{jn\omega_0 t} - e^{-jn\omega_0 t}}{2j}$$

삼각함수 급수에 대입하고 양수 및 음수 $n$에 대해 $e^{jn\omega_0 t}$를 가진 항들을 정리하면 복소 형태가 도출됩니다.

---

## 4. 푸리에 계수 계산

### 4.1 구형파(Square Wave)

주기 $T_0$와 진폭 $A$를 가진 **구형파(square wave)**:

$$x(t) = \begin{cases} A & 0 < t < T_0/2 \\ -A & T_0/2 < t < T_0 \end{cases}$$

**삼각함수 계수** (직접 계산):

$$a_0 = 0 \quad \text{(평균값 0)}$$

$$a_n = 0 \quad \text{(모든 } n \text{에 대해, 반파 대칭으로 인해)}$$

$$b_n = \begin{cases} \frac{4A}{n\pi} & n \text{이 홀수} \\ 0 & n \text{이 짝수} \end{cases}$$

따라서:

$$x(t) = \frac{4A}{\pi} \left[\sin(\omega_0 t) + \frac{1}{3}\sin(3\omega_0 t) + \frac{1}{5}\sin(5\omega_0 t) + \cdots \right]$$

**복소 계수**:

$$c_n = \begin{cases} \frac{-2jA}{n\pi} & n \text{이 홀수} \\ 0 & n \text{이 짝수 (}n = 0\text{ 포함)} \end{cases}$$

### 4.2 톱니파(Sawtooth Wave)

한 주기 $[0, T_0)$ 동안 $-A$에서 $A$로 선형적으로 증가하는 톱니파:

$$x(t) = A\left(\frac{2t}{T_0} - 1\right), \quad 0 \leq t < T_0$$

**계수**:

$$a_0 = 0, \quad a_n = 0$$

$$b_n = \frac{-2A}{n\pi}(-1)^{n+1} = \frac{2A}{n\pi}(-1)^{n+1} \cdot (-1) = \frac{(-1)^{n+1} \cdot 2A}{n\pi}$$

더 정확하게:

$$b_n = \frac{-2A}{n\pi} \quad \Rightarrow \quad x(t) = \frac{-2A}{\pi}\sum_{n=1}^{\infty} \frac{(-1)^n}{n} \sin(n\omega_0 t)$$

### 4.3 삼각파(Triangle Wave)

진폭 $A$와 주기 $T_0$를 가진 삼각파:

$$x(t) = \begin{cases} \frac{4A}{T_0}t & 0 \leq t \leq T_0/4 \\ A - \frac{4A}{T_0}(t - T_0/4) & T_0/4 \leq t \leq 3T_0/4 \\ -A + \frac{4A}{T_0}(t - 3T_0/4) & 3T_0/4 \leq t \leq T_0 \end{cases}$$

반파 대칭을 가진 우함수이므로:

$$a_n = \begin{cases} \frac{8A}{n^2\pi^2} & n = 1, 5, 9, \ldots \\ \frac{-8A}{n^2\pi^2} & n = 3, 7, 11, \ldots \\ 0 & n \text{이 짝수} \end{cases}$$

$$b_n = 0 \quad \text{모든 } n \text{에 대해}$$

콤팩트 형태:

$$x(t) = \frac{8A}{\pi^2} \sum_{k=0}^{\infty} \frac{(-1)^k}{(2k+1)^2} \cos((2k+1)\omega_0 t)$$

### 4.4 정류 정현파(Rectified Sinusoid, 전파 정류)

$$x(t) = |A\sin(\omega_0 t)|$$

이는 주기 $T_0/2$를 가지므로(기본 주파수 $2\omega_0$), 푸리에 급수는 원래 $\omega_0$의 짝수 고조파를 사용합니다:

$$x(t) = \frac{2A}{\pi} - \frac{4A}{\pi}\sum_{n=1}^{\infty} \frac{1}{4n^2 - 1} \cos(2n\omega_0 t)$$

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Fourier coefficients computation ---

def compute_fourier_coefficients(x_func, T0, N_harmonics, N_points=10000):
    """
    Compute complex Fourier coefficients numerically.

    Parameters:
        x_func: function of t returning signal values
        T0: fundamental period
        N_harmonics: number of harmonics (positive + negative)
        N_points: integration resolution

    Returns:
        n_values: harmonic indices
        cn: complex Fourier coefficients
    """
    omega0 = 2 * np.pi / T0
    t = np.linspace(0, T0, N_points, endpoint=False)
    dt = T0 / N_points
    x_vals = x_func(t)

    n_values = np.arange(-N_harmonics, N_harmonics + 1)
    cn = np.zeros(len(n_values), dtype=complex)

    for i, n in enumerate(n_values):
        cn[i] = np.mean(x_vals * np.exp(-1j * n * omega0 * t))

    return n_values, cn


# Square wave
T0 = 1.0
A = 1.0
def square_wave(t):
    return A * np.sign(np.sin(2 * np.pi * t / T0))

n_vals, cn = compute_fourier_coefficients(square_wave, T0, 20)

# Display non-negligible coefficients
print("Square Wave Fourier Coefficients (|cn| > 0.01):")
print(f"{'n':>4} | {'|cn|':>10} | {'angle(cn) (deg)':>16} | {'Expected |cn|':>14}")
for n, c in zip(n_vals, cn):
    if abs(c) > 0.01:
        expected = 2 * A / (abs(n) * np.pi) if n % 2 != 0 else 0
        print(f"{n:>4} | {abs(c):>10.6f} | {np.angle(c)*180/np.pi:>16.1f} | {expected:>14.6f}")
```

---

## 5. 푸리에 급수의 수렴

### 5.1 디리클레 조건(Dirichlet Conditions)

다음 **디리클레 조건**이 만족되면, $x(t)$가 연속인 모든 점에서 푸리에 급수는 $x(t)$로 수렴합니다:

1. $x(t)$는 한 주기에서 절대 적분 가능합니다: $\int_{T_0} |x(t)| \, dt < \infty$
2. $x(t)$는 한 주기 내에서 유한한 수의 극대값과 극소값을 가집니다
3. $x(t)$는 한 주기 내에서 유한한 수의 불연속점을 가집니다

불연속점에서, 푸리에 급수는 **중간값(midpoint)**으로 수렴합니다:

$$\text{FS}\{x(t_0)\} = \frac{x(t_0^+) + x(t_0^-)}{2}$$

### 5.2 수렴의 종류

**점별 수렴(Pointwise convergence)**: $\lim_{N \to \infty} S_N(t) = x(t)$, 각 특정 $t$에서 (불연속점을 제외할 수 있음).

**균등 수렴(Uniform convergence)**: $\lim_{N \to \infty} \max_t |x(t) - S_N(t)| = 0$. 이는 더 강한 조건이며 $x(t)$의 연속성을 필요로 합니다.

**평균제곱(L2) 수렴(Mean-square convergence)**: $\lim_{N \to \infty} \int_{T_0} |x(t) - S_N(t)|^2 \, dt = 0$. 이는 제곱 적분 가능한 신호에 대해 항상 성립합니다.

### 5.3 수렴 속도

푸리에 계수가 감소하는 속도는 신호의 매끄러움(smoothness)에 따라 달라집니다:

| 신호 특성 | 계수 감소율 | 예시 |
|-----------|------------|------|
| 불연속 | $|c_n| \sim 1/n$ | 구형파 |
| 연속이지만 도함수가 불연속 | $|c_n| \sim 1/n^2$ | 삼각파 |
| 연속이고 연속 도함수 보유 | $|c_n| \sim 1/n^3$ | 포물선 파 |
| $k$번 미분 가능 | $|c_n| \sim 1/n^{k+1}$ | 더 매끄러운 신호 |
| 무한히 미분 가능 | $1/n$의 임의 거듭제곱보다 빠름 | 가우시안 |

> **핵심**: 더 매끄러운 신호는 더 빠르게 감소하는 푸리에 계수를 가지므로, 좋은 근사를 위해 더 적은 항이 필요합니다. 불연속은 느린 ($1/n$) 감소를 야기하며, 이것이 깁스 현상으로 나타납니다.

---

## 6. 깁스 현상

### 6.1 설명

푸리에 급수가 점프 불연속을 가진 신호를 근사할 때, 부분 합 $S_N(t)$은 불연속점 근처에서 **링잉(ringing, 진동)** 현상을 보입니다. $N$이 증가할수록:

- 오버슈트/언더슈트의 폭은 좁아집니다 (불연속점에 더 가까이 이동)
- **최대 오버슈트는 $N$에 관계없이 점프 크기의 약 9%**를 유지합니다

이것이 J. Willard Gibbs의 이름을 딴 **깁스 현상(Gibbs phenomenon)**입니다.

### 6.2 수학적 설명

단위 계단 불연속(0에서 1로 점프)에 대해, 깁스 오버슈트는 다음으로 수렴합니다:

$$\frac{1}{\pi} \int_0^{\pi} \frac{\sin(u)}{u} \, du - \frac{1}{2} \approx 0.0895$$

이는 푸리에 부분 합의 최대값이 $0.5$ 대신 약 $\frac{1}{2} + 0.0895 = 0.5895$에 도달한다는 것을 의미하며, 점프의 약 **8.95%**의 오버슈트에 해당합니다.

### 6.3 실용적 시사점

- 깁스 현상은 푸리에 급수에 내재된 것으로, 더 많은 항을 추가해도 제거할 수 없습니다
- **윈도잉(windowing)**(예: Fejer 합산, Lanczos 시그마 인수)으로 줄일 수 있습니다
- 필터 설계에서는 **통과대역 리플(passband ripple)**로 나타납니다 (레슨 09 참조)
- 영상 처리에서는 날카로운 에지에서 **링잉 아티팩트(ringing artifacts)**를 유발합니다

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Gibbs phenomenon demonstration ---
T0 = 2 * np.pi
omega0 = 1.0
t = np.linspace(-np.pi, 3 * np.pi, 5000)

# Square wave Fourier series partial sums
def square_fourier_partial(t, N):
    """Partial sum of square wave Fourier series with N harmonics."""
    result = np.zeros_like(t)
    for k in range(1, N + 1):
        if k % 2 == 1:  # odd harmonics only
            result += (4 / (k * np.pi)) * np.sin(k * t)
    return result

# True square wave
x_true = np.sign(np.sin(t))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

N_values = [3, 9, 31, 101]
for ax, N in zip(axes.flat, N_values):
    S_N = square_fourier_partial(t, N)
    ax.plot(t / np.pi, x_true, 'b--', linewidth=1, alpha=0.5, label='Square wave')
    ax.plot(t / np.pi, S_N, 'r-', linewidth=1.5, label=f'$S_{{{N}}}(t)$')

    # Mark the Gibbs overshoot
    if N > 5:
        # Find max near discontinuity at t=0
        mask = (t > 0) & (t < np.pi / 2)
        max_val = np.max(S_N[mask])
        overshoot_pct = (max_val - 1.0) * 100
        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
        ax.annotate(f'Overshoot: {overshoot_pct:.1f}%',
                    xy=(0.15, max_val), fontsize=10, color='darkred')

    ax.set_title(f'N = {N} harmonics')
    ax.set_xlabel('$t / \\pi$')
    ax.set_ylabel('Amplitude')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-1.4, 1.4])
    ax.set_xlim([-0.5, 2.5])

plt.suptitle('Gibbs Phenomenon: Fourier Series of Square Wave', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('gibbs_phenomenon.png', dpi=150, bbox_inches='tight')
plt.show()

# Measure overshoot vs N
print("\n=== Gibbs Overshoot vs Number of Harmonics ===")
print(f"{'N':>6} | {'Max Value':>10} | {'Overshoot %':>12}")
print("-" * 35)
for N in [5, 11, 21, 51, 101, 201, 501, 1001]:
    t_fine = np.linspace(0.001, 0.5, 50000)
    S = square_fourier_partial(t_fine, N)
    max_val = np.max(S)
    overshoot = (max_val - 1.0) * 100
    print(f"{N:>6} | {max_val:>10.6f} | {overshoot:>11.4f}%")
```

---

## 7. 파르세발 정리

### 7.1 진술

**파르세발 정리(Parseval's theorem)**(푸리에 급수에 대한)는 주기 신호의 평균 전력이 푸리에 계수의 제곱 크기의 합과 같다고 진술합니다:

$$\frac{1}{T_0} \int_{T_0} |x(t)|^2 \, dt = \sum_{n=-\infty}^{\infty} |c_n|^2$$

삼각함수 계수로 표현하면:

$$\frac{1}{T_0} \int_{T_0} |x(t)|^2 \, dt = a_0^2 + \frac{1}{2}\sum_{n=1}^{\infty} (a_n^2 + b_n^2) = \sum_{n=0}^{\infty} \frac{C_n^2}{2 - \delta_{n0}}$$

### 7.2 해석

파르세발 정리는 **에너지 보존(conservation of energy)** 진술입니다:

- **좌변**: 시간 영역에서 계산된 총 평균 전력
- **우변**: 각 고조파의 전력 기여의 합

각 고조파 $n$은 총 전력에 $|c_n|^2 + |c_{-n}|^2 = 2|c_n|^2$ ($n \neq 0$)의 전력을 기여합니다.

### 7.3 예: 구형파 전력

진폭 $A$를 가진 구형파에 대해:

$$P_x = \frac{1}{T_0} \int_{T_0} A^2 \, dt = A^2$$

푸리에 계수로부터 (홀수 $n$에 대해 $|c_n| = 2A/(n\pi)$):

$$\sum_{n \text{ 홀수}} |c_n|^2 = 2 \sum_{k=0}^{\infty} \left(\frac{2A}{(2k+1)\pi}\right)^2 = \frac{8A^2}{\pi^2} \sum_{k=0}^{\infty} \frac{1}{(2k+1)^2} = \frac{8A^2}{\pi^2} \cdot \frac{\pi^2}{8} = A^2 \quad \checkmark$$

또한 이로부터 아름다운 항등식이 증명됩니다: $\sum_{k=0}^{\infty} \frac{1}{(2k+1)^2} = \frac{\pi^2}{8}$.

### 7.4 전력 스펙트럼

주기 신호의 **전력 스펙트럼(power spectrum)**은 주파수 $n\omega_0$에 대해 값 $\{|c_n|^2\}$를 나타낸 것입니다. 전력이 고조파에 어떻게 분포되는지 보여줍니다.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Parseval's theorem verification ---

T0 = 1.0
omega0 = 2 * np.pi / T0
A = 1.0

# Square wave
def square_wave(t):
    return A * np.sign(np.sin(2 * np.pi * t / T0))

t = np.linspace(0, T0, 100000, endpoint=False)
x = square_wave(t)

# Time-domain power
P_time = np.mean(x**2)

# Frequency-domain power (Parseval's)
N_max = 200
n_vals = np.arange(-N_max, N_max + 1)
cn = np.zeros(len(n_vals), dtype=complex)
for i, n in enumerate(n_vals):
    cn[i] = np.mean(x * np.exp(-1j * n * omega0 * t))

P_freq = np.sum(np.abs(cn)**2)

# Cumulative power contribution
cn_positive = cn[n_vals >= 0]
n_positive = n_vals[n_vals >= 0]
power_each = np.abs(cn_positive)**2
power_each[1:] *= 2  # double for n > 0 (conjugate symmetry)
P_cumulative = np.cumsum(power_each)

print(f"=== Parseval's Theorem Verification (Square Wave) ===")
print(f"Time-domain power:      P = {P_time:.8f}")
print(f"Frequency-domain power: P = {P_freq:.8f}")
print(f"Error: {abs(P_time - P_freq):.2e}")
print()

# How many harmonics capture 99% of power?
threshold = 0.99 * P_time
n_99 = n_positive[np.searchsorted(P_cumulative, threshold)]
print(f"Harmonics for 99% power: n = {n_99}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Power spectrum
axes[0].stem(n_positive[:30], power_each[:30] / P_time * 100,
             linefmt='b-', markerfmt='bo', basefmt='k-')
axes[0].set_title('Power Spectrum (% of Total Power)')
axes[0].set_xlabel('Harmonic number n')
axes[0].set_ylabel('Power contribution (%)')
axes[0].grid(True, alpha=0.3)

# Cumulative power
axes[1].plot(n_positive[:50], P_cumulative[:50] / P_time * 100, 'r-o', markersize=3)
axes[1].axhline(y=99, color='gray', linestyle='--', label='99%')
axes[1].axhline(y=95, color='lightgray', linestyle='--', label='95%')
axes[1].set_title("Cumulative Power (Parseval's)")
axes[1].set_xlabel('Number of harmonics included')
axes[1].set_ylabel('Cumulative power (%)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('parseval_theorem.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 8. 선 스펙트럼

### 8.1 정의

주기 신호의 스펙트럼은 **이산적(discrete)**(비주기 신호의 연속 스펙트럼과 대조적으로, $\omega_0$의 배수에서 선들의 집합)입니다 (레슨 04 참조).

**선 스펙트럼(line spectrum)**은 두 개의 그래프로 구성됩니다:

1. **진폭 스펙트럼(Amplitude spectrum)**: $|c_n|$ 대 $n\omega_0$ (또는 단순히 $n$에 대해)
2. **위상 스펙트럼(Phase spectrum)**: $\angle c_n$ 대 $n\omega_0$

실수 신호의 경우, 진폭 스펙트럼은 **우함수**($|c_n| = |c_{-n}|$)이고 위상 스펙트럼은 **기함수**($\angle c_n = -\angle c_{-n}$)입니다.

### 8.2 단측 대 양측 스펙트럼

| 종류 | 주파수 범위 | 사용하는 계수 |
|------|------------|-------------|
| **양측(Two-sided)** | $-\infty < n < \infty$ | 복소 $c_n$ |
| **단측(One-sided)** | $n \geq 0$ | 콤팩트 $C_n = 2|c_n|$ ($C_0 = |c_0|$) |

단측 스펙트럼은 주파수 $n\omega_0$에서의 에너지가 $c_n$과 $c_{-n}$ 사이에 분리되기 때문에 ($n > 0$에 대해) 복소 계수 크기의 두 배인 진폭을 사용합니다.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Line spectra for different waveforms ---

T0 = 1.0
omega0 = 2 * np.pi / T0
t = np.linspace(0, T0, 10000, endpoint=False)
N_harm = 20

waveforms = {
    'Square Wave': lambda t: np.sign(np.sin(2 * np.pi * t / T0)),
    'Sawtooth Wave': lambda t: 2 * (t / T0 - np.floor(t / T0 + 0.5)),
    'Triangle Wave': lambda t: 2 * np.abs(2 * (t / T0 - np.floor(t / T0 + 0.5))) - 1,
    'Half-Rectified Sine': lambda t: np.maximum(0, np.sin(2 * np.pi * t / T0)),
}

fig, axes = plt.subplots(len(waveforms), 3, figsize=(16, 3.5 * len(waveforms)))

for row, (name, func) in enumerate(waveforms.items()):
    x = func(t)

    # Compute complex Fourier coefficients
    n_range = np.arange(-N_harm, N_harm + 1)
    cn = np.zeros(len(n_range), dtype=complex)
    for i, n in enumerate(n_range):
        cn[i] = np.mean(x * np.exp(-1j * n * omega0 * t))

    # Time-domain signal
    axes[row, 0].plot(t, x, 'b-', linewidth=1.5)
    axes[row, 0].set_title(f'{name}')
    axes[row, 0].set_xlabel('t')
    axes[row, 0].set_ylabel('x(t)')
    axes[row, 0].grid(True, alpha=0.3)

    # Amplitude spectrum (two-sided)
    axes[row, 1].stem(n_range, np.abs(cn), linefmt='r-', markerfmt='ro', basefmt='k-')
    axes[row, 1].set_title(f'Amplitude Spectrum $|c_n|$')
    axes[row, 1].set_xlabel('Harmonic n')
    axes[row, 1].set_ylabel('$|c_n|$')
    axes[row, 1].grid(True, alpha=0.3)

    # Phase spectrum (two-sided)
    phase = np.angle(cn)
    # Zero out phase for negligible coefficients
    phase[np.abs(cn) < 1e-10] = 0
    axes[row, 2].stem(n_range, phase * 180 / np.pi, linefmt='g-', markerfmt='go',
                      basefmt='k-')
    axes[row, 2].set_title(f'Phase Spectrum $\\angle c_n$')
    axes[row, 2].set_xlabel('Harmonic n')
    axes[row, 2].set_ylabel('Phase (degrees)')
    axes[row, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('line_spectra.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 9. 응용: 표준 파형 분해

### 9.1 푸리에 급수 근사 갤러리

푸리에 급수가 표준 파형을 점진적으로 재구성하는 방식을 시각화해 봅시다.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Progressive Fourier reconstruction ---

T0 = 2 * np.pi
omega0 = 1.0
t = np.linspace(-np.pi, 3 * np.pi, 2000)

def fourier_square(t, N):
    """Square wave: sum of (4/n*pi)*sin(n*t) for odd n."""
    result = np.zeros_like(t)
    for n in range(1, N + 1, 2):
        result += (4 / (n * np.pi)) * np.sin(n * t)
    return result

def fourier_sawtooth(t, N):
    """Sawtooth: sum of (-2/n*pi)*(-1)^n * sin(n*t)."""
    result = np.zeros_like(t)
    for n in range(1, N + 1):
        result += (2 / (n * np.pi)) * ((-1)**(n + 1)) * np.sin(n * t)
    return result

def fourier_triangle(t, N):
    """Triangle: sum of (8/n^2*pi^2)*(-1)^k * cos(n*t) for odd n."""
    result = np.zeros_like(t)
    k = 0
    for n in range(1, N + 1, 2):
        result += (8 / (n**2 * np.pi**2)) * ((-1)**k) * np.cos(n * t)
        k += 1
    return result

# True waveforms
square_true = np.sign(np.sin(t))
sawtooth_true = (t + np.pi) % (2 * np.pi) / np.pi - 1
triangle_true = 2 * np.abs(2 * ((t + np.pi) / (2 * np.pi) - np.floor((t + np.pi) / (2 * np.pi) + 0.5))) - 1

waveforms = [
    ('Square Wave', square_true, fourier_square),
    ('Sawtooth Wave', sawtooth_true, fourier_sawtooth),
    ('Triangle Wave', triangle_true, fourier_triangle),
]

N_terms_list = [1, 3, 7, 21]

fig, axes = plt.subplots(len(waveforms), len(N_terms_list), figsize=(18, 10))

for row, (name, x_true, fourier_func) in enumerate(waveforms):
    for col, N in enumerate(N_terms_list):
        x_approx = fourier_func(t, N)
        axes[row, col].plot(t / np.pi, x_true, 'b--', linewidth=1, alpha=0.4,
                           label='True')
        axes[row, col].plot(t / np.pi, x_approx, 'r-', linewidth=1.5,
                           label=f'N={N}')
        axes[row, col].set_ylim([-1.5, 1.5])
        axes[row, col].set_xlim([-0.5, 2.5])
        if row == 0:
            axes[row, col].set_title(f'N = {N} harmonics')
        if col == 0:
            axes[row, col].set_ylabel(name)
        axes[row, col].legend(fontsize=8)
        axes[row, col].grid(True, alpha=0.3)

plt.suptitle('Fourier Series Convergence for Standard Waveforms', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('fourier_convergence.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 9.2 물리적 응용: 열전도

푸리에 급수는 원래 Joseph Fourier가 열 방정식(heat equation)을 풀기 위해 개발했습니다. 초기 온도 분포 $f(x)$를 가지고 양 끝이 0도로 유지된 길이 $L$의 금속 막대를 생각해 봅시다.

위치 $x$와 시간 $t$에서의 온도는:

$$u(x, t) = \sum_{n=1}^{\infty} b_n \sin\left(\frac{n\pi x}{L}\right) e^{-\alpha (n\pi/L)^2 t}$$

여기서 $b_n$은 $f(x)$의 푸리에 사인 급수 계수이고 $\alpha$는 열확산율(thermal diffusivity)입니다.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Heat equation solution using Fourier series ---

L = 1.0          # rod length
alpha = 0.01     # thermal diffusivity
N_terms = 50     # Fourier terms

# Initial temperature: step function (hot in the middle)
def initial_temp(x):
    return np.where((x > 0.25) & (x < 0.75), 1.0, 0.0)

# Compute Fourier sine coefficients
x_int = np.linspace(0, L, 10000)
dx = x_int[1] - x_int[0]
f_x = initial_temp(x_int)

bn = np.zeros(N_terms + 1)
for n in range(1, N_terms + 1):
    bn[n] = (2 / L) * np.trapz(f_x * np.sin(n * np.pi * x_int / L), x_int)

# Solution at various times
x = np.linspace(0, L, 500)
times = [0, 0.5, 2, 5, 10, 20]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for ax, t_val in zip(axes.flat, times):
    u = np.zeros_like(x)
    for n in range(1, N_terms + 1):
        u += bn[n] * np.sin(n * np.pi * x / L) * np.exp(-alpha * (n * np.pi / L)**2 * t_val)

    ax.plot(x, u, 'r-', linewidth=2)
    ax.fill_between(x, u, alpha=0.2, color='red')
    ax.set_title(f't = {t_val}')
    ax.set_xlabel('Position x')
    ax.set_ylabel('Temperature u(x, t)')
    ax.set_ylim([-0.1, 1.1])
    ax.grid(True, alpha=0.3)

plt.suptitle('Heat Equation: Fourier Series Solution', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('heat_equation_fourier.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 9.3 신호 합성: 가산 사운드 합성

푸리에 급수는 오디오 공학에서 **가산 합성(additive synthesis)**의 이론적 기반입니다. 이는 순수한 음(정현파)을 더하여 복잡한 소리를 만드는 방식입니다.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Additive sound synthesis ---

fs = 44100  # CD quality sampling rate
duration = 0.5
t = np.arange(int(fs * duration)) / fs

# Synthesize different timbres at A4 (440 Hz)
f0 = 440

def synthesize(harmonics, amplitudes, phases=None):
    """Additive synthesis from harmonic specification."""
    if phases is None:
        phases = np.zeros(len(harmonics))
    signal = np.zeros_like(t)
    for n, amp, phi in zip(harmonics, amplitudes, phases):
        signal += amp * np.sin(2 * np.pi * n * f0 * t + phi)
    # Normalize
    signal /= np.max(np.abs(signal))
    return signal

# Different timbres
timbres = {
    'Pure Tone (1 harmonic)': synthesize([1], [1.0]),
    'Clarinet-like (odd harmonics)': synthesize(
        [1, 3, 5, 7, 9, 11],
        [1.0, 0.75, 0.5, 0.14, 0.5, 0.12]
    ),
    'Bright/Sawtooth (all harmonics)': synthesize(
        range(1, 16),
        [1.0 / n for n in range(1, 16)]
    ),
    'Organ-like (specific harmonics)': synthesize(
        [1, 2, 3, 4, 6, 8],
        [1.0, 0.5, 0.3, 0.25, 0.1, 0.05]
    ),
}

fig, axes = plt.subplots(len(timbres), 2, figsize=(16, 3 * len(timbres)))

for row, (name, signal) in enumerate(timbres.items()):
    # Time domain (show 3 cycles)
    n_show = int(3 * fs / f0)
    axes[row, 0].plot(t[:n_show] * 1000, signal[:n_show], 'b-', linewidth=1.5)
    axes[row, 0].set_title(f'{name} — Time Domain')
    axes[row, 0].set_xlabel('Time (ms)')
    axes[row, 0].set_ylabel('Amplitude')
    axes[row, 0].grid(True, alpha=0.3)

    # Frequency domain
    N = len(signal)
    freqs = np.fft.rfftfreq(N, 1 / fs)
    spectrum = np.abs(np.fft.rfft(signal)) / N * 2
    axes[row, 1].stem(freqs[:30 * len(t) // fs],
                      spectrum[:30 * len(t) // fs],
                      linefmt='r-', markerfmt='ro', basefmt='k-')
    axes[row, 1].set_title(f'{name} — Spectrum')
    axes[row, 1].set_xlabel('Frequency (Hz)')
    axes[row, 1].set_ylabel('Amplitude')
    axes[row, 1].set_xlim([0, 8000])
    axes[row, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('additive_synthesis.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 10. 이산시간 푸리에 급수

### 10.1 정의

주기 $N$을 가진 이산시간 주기 신호 $x[n]$은 다음과 같이 표현할 수 있습니다:

$$x[n] = \sum_{k=0}^{N-1} c_k \, e^{j(2\pi/N)kn}$$

여기서 **이산시간(DT) 푸리에 급수 계수**는:

$$c_k = \frac{1}{N} \sum_{n=0}^{N-1} x[n] \, e^{-j(2\pi/N)kn}, \quad k = 0, 1, \ldots, N-1$$

### 10.2 연속시간 푸리에 급수와의 주요 차이점

| 특성 | 연속시간(Continuous-Time) | 이산시간(Discrete-Time) |
|------|--------------------------|------------------------|
| 고조파 수 | 무한 | 유한 ($N$개) |
| 계수 인덱스 범위 | $n \in \mathbb{Z}$ (무한) | $k = 0, 1, \ldots, N-1$ (유한) |
| 급수 | 근사 (절사된) | **정확한** 표현 |
| 수렴 문제 | 깁스 현상, 디리클레 조건 | 없음 (유한 합) |

이산시간 푸리에 급수는 주기 $N$을 가진 주기 수열이 $N$개의 독립 값만 가지기 때문에, $N$개의 계수로 항상 정확합니다.

### 10.3 DFT와의 연결

이산시간 푸리에 급수 계수는 본질적으로 한 주기의 **이산 푸리에 변환(Discrete Fourier Transform, DFT)**을 $1/N$로 스케일한 것입니다. 이 연결은 레슨 06의 핵심입니다.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Discrete-time Fourier series ---

N = 16  # period
n = np.arange(N)

# Example: discrete-time square wave
x = np.ones(N)
x[N//2:] = -1

# DT Fourier coefficients
ck = np.zeros(N, dtype=complex)
for k in range(N):
    ck[k] = (1 / N) * np.sum(x * np.exp(-1j * 2 * np.pi * k * n / N))

# Reconstruction (should be exact)
x_reconstructed = np.zeros(N, dtype=complex)
for k in range(N):
    x_reconstructed += ck[k] * np.exp(1j * 2 * np.pi * k * n / N)

print("Original:      ", x)
print("Reconstructed: ", np.real(x_reconstructed).round(10))
print("Max error:     ", np.max(np.abs(x - np.real(x_reconstructed))))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].stem(n, x, linefmt='b-', markerfmt='bo', basefmt='k-')
axes[0].set_title(f'DT Signal $x[n]$ (period N={N})')
axes[0].set_xlabel('n')
axes[0].grid(True, alpha=0.3)

axes[1].stem(np.arange(N), np.abs(ck), linefmt='r-', markerfmt='ro', basefmt='k-')
axes[1].set_title('Amplitude $|c_k|$')
axes[1].set_xlabel('k')
axes[1].grid(True, alpha=0.3)

axes[2].stem(np.arange(N), np.angle(ck) * 180 / np.pi, linefmt='g-', markerfmt='go',
             basefmt='k-')
axes[2].set_title('Phase $\\angle c_k$ (degrees)')
axes[2].set_xlabel('k')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dt_fourier_series.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 11. Python 예제

### 11.1 종합 푸리에 급수 분석기

```python
import numpy as np
import matplotlib.pyplot as plt

class FourierSeriesAnalyzer:
    """Complete Fourier series analysis toolkit."""

    def __init__(self, signal_func, T0, name="Signal"):
        self.signal_func = signal_func
        self.T0 = T0
        self.omega0 = 2 * np.pi / T0
        self.name = name
        self.t = np.linspace(0, T0, 10000, endpoint=False)
        self.x = signal_func(self.t)

    def compute_coefficients(self, N_max):
        """Compute complex Fourier coefficients c_n for |n| <= N_max."""
        n_range = np.arange(-N_max, N_max + 1)
        cn = np.zeros(len(n_range), dtype=complex)
        for i, n in enumerate(n_range):
            cn[i] = np.mean(self.x * np.exp(-1j * n * self.omega0 * self.t))
        return n_range, cn

    def reconstruct(self, t, n_range, cn):
        """Reconstruct signal from Fourier coefficients."""
        x_approx = np.zeros_like(t, dtype=complex)
        for n, c in zip(n_range, cn):
            x_approx += c * np.exp(1j * n * self.omega0 * t)
        return np.real(x_approx)

    def analyze(self, N_max=30, N_show=[1, 3, 7, 15, 30]):
        """Complete analysis with plots."""
        n_range, cn = self.compute_coefficients(N_max)

        fig = plt.figure(figsize=(16, 14))

        # 1. Original signal
        ax1 = fig.add_subplot(3, 2, 1)
        t_plot = np.linspace(-self.T0/2, 1.5*self.T0, 3000)
        ax1.plot(t_plot / self.T0, self.signal_func(t_plot), 'b-', linewidth=2)
        ax1.set_title(f'{self.name} — Time Domain')
        ax1.set_xlabel('t / T₀')
        ax1.set_ylabel('x(t)')
        ax1.grid(True, alpha=0.3)

        # 2. Amplitude spectrum
        ax2 = fig.add_subplot(3, 2, 2)
        ax2.stem(n_range, np.abs(cn), linefmt='r-', markerfmt='ro', basefmt='k-')
        ax2.set_title('Amplitude Spectrum $|c_n|$')
        ax2.set_xlabel('Harmonic n')
        ax2.set_ylabel('$|c_n|$')
        ax2.grid(True, alpha=0.3)

        # 3. Progressive reconstruction
        ax3 = fig.add_subplot(3, 2, 3)
        t_recon = np.linspace(0, 2 * self.T0, 2000)
        ax3.plot(t_recon / self.T0, self.signal_func(t_recon), 'k--',
                linewidth=1, alpha=0.4, label='True')
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(N_show)))
        for N, color in zip(N_show, colors):
            n_sub = np.arange(-N, N + 1)
            cn_sub = cn[(n_range >= -N) & (n_range <= N)]
            x_approx = self.reconstruct(t_recon, n_sub, cn_sub)
            ax3.plot(t_recon / self.T0, x_approx, color=color,
                    linewidth=1, label=f'N={N}')
        ax3.set_title('Progressive Reconstruction')
        ax3.set_xlabel('t / T₀')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # 4. Reconstruction error vs N
        ax4 = fig.add_subplot(3, 2, 4)
        N_test = np.arange(1, N_max + 1)
        errors = []
        for N in N_test:
            n_sub = np.arange(-N, N + 1)
            cn_sub = cn[(n_range >= -N) & (n_range <= N)]
            x_approx = self.reconstruct(self.t, n_sub, cn_sub)
            mse = np.mean((self.x - x_approx)**2)
            errors.append(mse)
        ax4.semilogy(N_test, errors, 'b-o', markersize=3)
        ax4.set_title('Mean Squared Error vs N')
        ax4.set_xlabel('Number of harmonics N')
        ax4.set_ylabel('MSE')
        ax4.grid(True, alpha=0.3)

        # 5. Power spectrum
        ax5 = fig.add_subplot(3, 2, 5)
        P_total = np.mean(self.x**2)
        power_contributions = np.abs(cn)**2
        n_pos = n_range[n_range >= 0]
        cn_pos = cn[n_range >= 0]
        P_cumulative = np.cumsum(np.abs(cn_pos)**2)
        # Add symmetric part
        for i, n in enumerate(n_pos):
            if n > 0:
                P_cumulative[i:] += np.abs(cn[n_range == -n])[0]**2
        ax5.plot(n_pos, P_cumulative / P_total * 100, 'r-o', markersize=3)
        ax5.axhline(y=99, color='gray', linestyle='--', label='99%')
        ax5.set_title("Cumulative Power (Parseval's)")
        ax5.set_xlabel('Max harmonic')
        ax5.set_ylabel('% of total power')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Phase spectrum
        ax6 = fig.add_subplot(3, 2, 6)
        phase = np.angle(cn)
        phase[np.abs(cn) < 1e-10] = 0
        ax6.stem(n_range, phase * 180 / np.pi, linefmt='g-', markerfmt='go',
                basefmt='k-')
        ax6.set_title('Phase Spectrum $\\angle c_n$ (degrees)')
        ax6.set_xlabel('Harmonic n')
        ax6.set_ylabel('Phase (°)')
        ax6.grid(True, alpha=0.3)

        plt.suptitle(f'Fourier Series Analysis: {self.name}', fontsize=14, y=1.01)
        plt.tight_layout()
        plt.savefig(f'fourier_analysis_{self.name.replace(" ", "_")}.png',
                    dpi=150, bbox_inches='tight')
        plt.show()


# Analyze different waveforms
T0 = 1.0

# Square wave
analyzer = FourierSeriesAnalyzer(
    lambda t: np.sign(np.sin(2 * np.pi * t / T0)),
    T0, "Square Wave"
)
analyzer.analyze()

# Triangle wave
analyzer = FourierSeriesAnalyzer(
    lambda t: 2 * np.abs(2 * (t / T0 - np.floor(t / T0 + 0.5))) - 1,
    T0, "Triangle Wave"
)
analyzer.analyze()
```

### 11.2 깁스 현상 완화

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Gibbs phenomenon mitigation using sigma factors ---

T0 = 2 * np.pi
t = np.linspace(-0.5, 2.5 * np.pi, 2000)
N = 30

# Square wave Fourier coefficients
def cn_square(n):
    if n == 0:
        return 0
    if n % 2 == 0:
        return 0
    return -2j / (n * np.pi)

# Standard partial sum
def partial_sum(t, N, window_func=None):
    result = np.zeros_like(t, dtype=complex)
    for n in range(-N, N + 1):
        c = cn_square(n)
        if window_func is not None:
            c *= window_func(n, N)
        result += c * np.exp(1j * n * t)
    return np.real(result)

# Sigma factors for Gibbs mitigation
def lanczos_sigma(n, N):
    """Lanczos sigma factor: sinc(n/N)."""
    if n == 0:
        return 1.0
    return np.sinc(n / N)

def fejer_sigma(n, N):
    """Fejer (Cesaro) kernel: 1 - |n|/N."""
    return max(0, 1 - abs(n) / N)

def raised_cosine_sigma(n, N):
    """Raised cosine (Hanning-like)."""
    return 0.5 * (1 + np.cos(np.pi * n / N))

# Compare methods
x_true = np.sign(np.sin(t))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

methods = [
    ("Standard Fourier (N=30)", None),
    ("Lanczos sigma factors", lanczos_sigma),
    ("Fejer (Cesaro) summation", fejer_sigma),
    ("Raised cosine window", raised_cosine_sigma),
]

for ax, (name, sigma) in zip(axes.flat, methods):
    S = partial_sum(t, N, sigma)
    ax.plot(t / np.pi, x_true, 'b--', linewidth=1, alpha=0.4, label='True')
    ax.plot(t / np.pi, S, 'r-', linewidth=1.5, label=name)
    ax.set_ylim([-1.4, 1.4])
    ax.set_xlim([-0.15, 2.2])
    ax.set_title(name)
    ax.set_xlabel('$t / \\pi$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Gibbs Phenomenon Mitigation Methods', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('gibbs_mitigation.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 11.3 대화형 계수 탐색기

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Effect of modifying individual Fourier coefficients ---

T0 = 1.0
omega0 = 2 * np.pi / T0
t = np.linspace(0, 2 * T0, 1000)
N_harm = 10

# Start with square wave coefficients
cn_original = np.zeros(2 * N_harm + 1, dtype=complex)
n_range = np.arange(-N_harm, N_harm + 1)

for i, n in enumerate(n_range):
    if n != 0 and n % 2 != 0:
        cn_original[i] = -2j / (n * np.pi)

def reconstruct(cn, t, n_range, omega0):
    x = np.zeros_like(t, dtype=complex)
    for c, n in zip(cn, n_range):
        x += c * np.exp(1j * n * omega0 * t)
    return np.real(x)

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# Original
x_orig = reconstruct(cn_original, t, n_range, omega0)
axes[0, 0].plot(t / T0, x_orig, 'b-', linewidth=2)
axes[0, 0].set_title('Original Square Wave (N=10)')
axes[0, 0].grid(True, alpha=0.3)

# Remove 3rd harmonic
cn_no3 = cn_original.copy()
cn_no3[n_range == 3] = 0
cn_no3[n_range == -3] = 0
x_no3 = reconstruct(cn_no3, t, n_range, omega0)
axes[0, 1].plot(t / T0, x_orig, 'b--', alpha=0.3, label='Original')
axes[0, 1].plot(t / T0, x_no3, 'r-', linewidth=2, label='No 3rd harmonic')
axes[0, 1].set_title('Remove 3rd Harmonic')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Double the 3rd harmonic
cn_double3 = cn_original.copy()
cn_double3[n_range == 3] *= 2
cn_double3[n_range == -3] *= 2
x_double3 = reconstruct(cn_double3, t, n_range, omega0)
axes[1, 0].plot(t / T0, x_orig, 'b--', alpha=0.3, label='Original')
axes[1, 0].plot(t / T0, x_double3, 'g-', linewidth=2, label='3rd harmonic x2')
axes[1, 0].set_title('Double the 3rd Harmonic')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Add even harmonics
cn_even = cn_original.copy()
for n in [2, 4, 6]:
    idx_pos = np.where(n_range == n)[0][0]
    idx_neg = np.where(n_range == -n)[0][0]
    cn_even[idx_pos] = 0.3 / n
    cn_even[idx_neg] = 0.3 / n
x_even = reconstruct(cn_even, t, n_range, omega0)
axes[1, 1].plot(t / T0, x_orig, 'b--', alpha=0.3, label='Original')
axes[1, 1].plot(t / T0, x_even, 'm-', linewidth=2, label='+ even harmonics')
axes[1, 1].set_title('Add Even Harmonics (breaks half-wave symmetry)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Phase shift all harmonics by pi/4
cn_phased = cn_original * np.exp(1j * np.pi / 4 * np.abs(n_range))
x_phased = reconstruct(cn_phased, t, n_range, omega0)
axes[2, 0].plot(t / T0, x_orig, 'b--', alpha=0.3, label='Original')
axes[2, 0].plot(t / T0, x_phased, 'orange', linewidth=2, label='Phase shifted')
axes[2, 0].set_title('Phase Shift Each Harmonic by $n \\cdot \\pi/4$')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

# Random phases (same magnitude)
np.random.seed(42)
cn_random_phase = np.abs(cn_original) * np.exp(1j * 2 * np.pi * np.random.rand(len(cn_original)))
# Keep conjugate symmetry for real output
for i, n in enumerate(n_range):
    if n < 0:
        cn_random_phase[i] = np.conj(cn_random_phase[n_range == -n][0])
x_random = reconstruct(cn_random_phase, t, n_range, omega0)
axes[2, 1].plot(t / T0, x_orig, 'b--', alpha=0.3, label='Original')
axes[2, 1].plot(t / T0, x_random, 'cyan', linewidth=2, label='Random phases')
axes[2, 1].set_title('Random Phases (same magnitudes)')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

for ax in axes.flat:
    ax.set_xlabel('$t / T_0$')
    ax.set_ylim([-1.8, 1.8])

plt.suptitle('Effect of Modifying Fourier Coefficients', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('coefficient_explorer.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 12. 요약

### 주요 공식

| 형태 | 급수 | 계수 |
|------|------|------|
| 삼각함수 | $x(t) = a_0 + \sum_{n=1}^{\infty}[a_n\cos(n\omega_0 t) + b_n\sin(n\omega_0 t)]$ | $a_0 = \frac{1}{T_0}\int x \, dt$, $a_n = \frac{2}{T_0}\int x\cos(n\omega_0 t) \, dt$, $b_n = \frac{2}{T_0}\int x\sin(n\omega_0 t) \, dt$ |
| 복소 지수 | $x(t) = \sum_{n=-\infty}^{\infty} c_n e^{jn\omega_0 t}$ | $c_n = \frac{1}{T_0}\int x(t) e^{-jn\omega_0 t} \, dt$ |
| 콤팩트 | $x(t) = C_0 + \sum_{n=1}^{\infty} C_n\cos(n\omega_0 t + \phi_n)$ | $C_n = \sqrt{a_n^2 + b_n^2}$, $\phi_n = -\arctan(b_n/a_n)$ |

### 개념 계층 구조

```
            주기 신호 x(t)
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  디리클레 조건            푸리에 계수
  (수렴 검증)             cn 또는 (an, bn)
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
              선 스펙트럼  파르세발    재구성
              |cn| vs n   전력 합     부분 합
                                        깁스 현상
```

### 핵심 정리

1. **푸리에 급수**는 주기 신호를 고조파적으로 관련된 정현파로 분해합니다
2. 삼각함수의 **직교성(orthogonality)**이 수학적 기반입니다
3. **복소 지수 형태**가 신호처리에서 선호됩니다 (LTI 시스템의 고유함수)
4. 계수 **감소율**은 신호의 매끄러움을 반영합니다 (불연속에서 $1/n$, 연속에서 $1/n^2$)
5. **깁스 현상**: 불연속점에서 9% 오버슈트는 더 많은 항을 추가해도 제거할 수 없습니다
6. **파르세발 정리**: 전력은 시간 영역과 주파수 영역 사이에서 보존됩니다
7. **선 스펙트럼**은 주기 신호의 완전한 주파수 영역 그림을 제공합니다

---

## 13. 연습 문제

### 연습 문제 1: 푸리에 계수 계산

다음 신호들에 대한 푸리에 급수 계수(삼각함수 및 복소 지수 모두)를 계산하세요:

1. $x(t) = |\sin(\omega_0 t)|$ (전파 정류 사인)
2. $x(t) = \cos^2(\omega_0 t)$ (힌트: 먼저 삼각함수 항등식 사용)
3. 펄스 열(pulse train): $x(t) = 1$ ($|t| < \tau/2$), $x(t) = 0$ ($\tau/2 < |t| < T_0/2$), 주기 $T_0$로 주기적

### 연습 문제 2: 대칭성 활용

각 신호에 대해 대칭 유형(우함수, 기함수, 반파)을 파악하고, 계산하지 않고 어떤 푸리에 계수가 0인지 결정하세요:

1. $[-\pi, \pi]$에서 $x(t) = t^2$, 주기적
2. $[-\pi, \pi]$에서 $x(t) = t$, 주기적
3. $[-\pi, \pi]$에서 $x(t) = |t|$, 주기적
4. 계단함수: $0 < t < \pi$에서 $x(t) = +1$, $-\pi < t < 0$에서 $x(t) = -1$

### 연습 문제 3: 깁스 현상 탐구

1. $N = 5, 21, 101$에 대해 구형파의 푸리에 부분 합 $S_N(t)$를 계산하고 그래프로 나타내세요
2. 각 $N$에 대해 최대 오버슈트를 수치적으로 구하세요
3. 오버슈트 비율이 약 8.95%로 수렴함을 보이세요
4. Lanczos 시그마 인수를 구현하고 줄어든 오버슈트를 시연하세요
5. 표준 푸리에 대 Lanczos 윈도우 부분 합의 MSE를 비교하세요

### 연습 문제 4: 파르세발 정리 응용

1. 파르세발 정리를 이용하여 톱니파에서 $\sum_{n=1}^{\infty} \frac{1}{n^2}$를 계산하세요
2. 파르세발 정리를 이용하여 삼각파에서 $\sum_{k=0}^{\infty} \frac{1}{(2k+1)^4}$를 계산하세요
3. 듀티 사이클 $d = \tau/T_0$를 가진 펄스 열에 대해:
   - $d$의 함수로 전력을 계산하세요
   - $d = 0.1, 0.25, 0.5$에 대해 전력 스펙트럼을 그리세요
   - 각 경우에 대해 파르세발 정리를 수치적으로 검증하세요

### 연습 문제 5: 신호 재구성 도전

$T_0 = 1$ s인 주기 신호의 진폭 스펙트럼 $|c_n|$과 위상 스펙트럼 $\angle c_n$ ($|n| \leq 10$)만 주어진 경우:

| $n$ | $|c_n|$ | $\angle c_n$ (rad) |
|-----|---------|-------------------|
| 0 | 0.5 | 0 |
| 1 | 0.8 | $-\pi/4$ |
| 2 | 0.3 | $-\pi/2$ |
| 3 | 0.6 | $\pi/3$ |
| 5 | 0.2 | $-\pi/6$ |
| 7 | 0.1 | $\pi/4$ |

1. $x(t)$를 재구성하고 그래프로 나타내세요
2. 파르세발 정리를 사용하여 $x(t)$의 평균 전력을 계산하세요
3. 모든 위상을 0으로 설정하면 신호 모양은 어떻게 변하나요?
4. 모든 위상을 무작위로 설정하면 어떻게 되나요?

### 연습 문제 6: 실제 신호의 푸리에 급수

1. 100 Hz의 톱니파를 44100 Hz로 0.1초 동안 샘플링하여 생성하세요
2. 해석적 공식과 수치 적분 모두를 사용하여 푸리에 계수를 계산하세요
3. 두 결과를 비교하세요
4. 첫 5, 10, 20개의 고조파만을 사용하여 신호를 재구성하세요
5. 원본과 비교한 각 재구성의 SNR(dB)을 계산하세요

### 연습 문제 7: 푸리에 급수와 LTI 시스템

주기 입력 $x(t) = \sum c_n e^{jn\omega_0 t}$가 주파수 응답 $H(j\omega)$를 가진 LTI 시스템에 인가됩니다.

1. 출력도 동일한 주기를 가진 주기 신호임을 보이세요
2. 출력 푸리에 계수가 $d_n = c_n \cdot H(jn\omega_0)$임을 보이세요
3. $f_0 = 100$ Hz인 구형파가 차단 주파수 500 Hz의 RC 저역통과 필터를 통과합니다. 출력 푸리에 계수를 계산하고 출력 신호를 재구성하세요.
4. 필터가 깁스 현상에 어떤 영향을 미치나요?

### 연습 문제 8: 이산시간 푸리에 급수

1. 이산 구형파를 나타내는 주기 $N = 32$의 이산시간 주기 수열 $x[n]$을 생성하세요
2. 모든 $N = 32$개의 이산시간 푸리에 급수 계수를 계산하세요
3. 재구성이 정확함을 검증하세요
4. `np.fft.fft`를 사용하여 계산한 DFT와 비교하세요
5. 이산시간 푸리에 급수에 깁스 현상이 없음을 보이세요 (이유 설명)

---

## 14. 참고 문헌

1. Oppenheim, A. V. & Willsky, A. S. *Signals and Systems* (2nd ed.), Ch. 3-4. Prentice Hall, 1997.
2. Haykin, S. & Van Veen, B. *Signals and Systems* (2nd ed.), Ch. 3, 6. Wiley, 2003.
3. Lathi, B. P. & Green, R. A. *Linear Systems and Signals* (3rd ed.), Ch. 6. Oxford University Press, 2018.
4. Boas, M. L. *Mathematical Methods in the Physical Sciences* (3rd ed.), Ch. 7. Wiley, 2006.
5. Smith, S. W. *The Scientist and Engineer's Guide to Digital Signal Processing*, Ch. 13. California Technical Publishing, 1997.

---

[이전: 02. LTI 시스템과 합성곱](./02_LTI_Systems_and_Convolution.md) | [다음: 04. 연속 푸리에 변환](./04_Continuous_Fourier_Transform.md) | [개요](./00_Overview.md)
