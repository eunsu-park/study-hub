# LTI 시스템과 합성곱

**이전**: [01. 신호와 시스템](./01_Signals_and_Systems.md) | **다음**: [03. 푸리에 급수와 응용](./03_Fourier_Series_and_Applications.md)

---

레슨 01에서 우리는 신호를 표현하는 언어와 시스템의 특성을 나타내는 속성들을 정립했습니다. 가능한 모든 시스템 중에서 중요성 면에서 가장 두드러지는 부류가 있습니다: **선형 시불변(LTI, Linear Time-Invariant) 시스템**입니다. 모든 LTI 시스템의 동작은 단 하나의 함수인 **임펄스 응답(impulse response)**에 의해 완전히 결정되며, 임의의 입력에 대한 출력은 **합성곱(convolution)**을 통해 계산할 수 있습니다. 이 레슨에서는 이러한 개념들을 기초부터 전개하고, 합성곱이 신호 처리에서 핵심 연산인 이유를 설명합니다.

**난이도**: ⭐⭐⭐

**학습 목표**:
- LTI 시스템이 신호 처리에서 중요한 이유를 설명할 수 있다
- 중첩 원리로부터 합성곱 적분과 합성곱 합을 유도할 수 있다
- 합성곱을 해석적으로 및 수치적으로 계산할 수 있다
- 합성곱의 성질(교환법칙, 결합법칙, 분배법칙)을 적용할 수 있다
- 계단 응답(step response)과 임펄스 응답의 관계를 파악할 수 있다
- 직렬, 병렬, 피드백 시스템 상호연결을 분석할 수 있다
- 임펄스 응답으로부터 BIBO 안정성을 판별할 수 있다
- LTI 시스템의 주파수 응답을 계산할 수 있다

---

## 목차

1. [왜 LTI 시스템인가?](#1-왜-lti-시스템인가)
2. [임펄스 응답](#2-임펄스-응답)
3. [합성곱 합 (이산 시간)](#3-합성곱-합-이산-시간)
4. [합성곱 적분 (연속 시간)](#4-합성곱-적분-연속-시간)
5. [합성곱 계산](#5-합성곱-계산)
6. [합성곱의 성질](#6-합성곱의-성질)
7. [계단 응답](#7-계단-응답)
8. [시스템 상호연결](#8-시스템-상호연결)
9. [LTI 시스템 안정성](#9-lti-시스템-안정성)
10. [LTI 시스템의 주파수 응답](#10-lti-시스템의-주파수-응답)
11. [Python 예제](#11-python-예제)
12. [요약](#12-요약)
13. [연습 문제](#13-연습-문제)
14. [참고문헌](#14-참고문헌)

---

## 1. 왜 LTI 시스템인가?

### 1.1 선형성과 시불변성의 힘

LTI 시스템은 두 가지 속성을 동시에 만족합니다:

**선형성(Linearity)** (중첩 원리):

$$\mathcal{T}\{a x_1(t) + b x_2(t)\} = a \mathcal{T}\{x_1(t)\} + b \mathcal{T}\{x_2(t)\}$$

**시불변성(Time Invariance)** (이동 불변성):

$$\text{만약 } x(t) \to y(t) \text{ 이면, } x(t - t_0) \to y(t - t_0)$$

이 두 성질이 합쳐지면 놀라운 능력이 생깁니다: 단 하나의 단순한 입력(임펄스)에 대한 시스템 응답만 알면, **임의의** 입력에 대한 응답을 구할 수 있습니다. 그 이유는:

1. **선형성**을 통해 임의의 입력을 기본 성분의 가중합으로 분해할 수 있습니다
2. **시불변성**은 시스템이 각 이동된 성분에 동일한 방식으로 응답함을 보장합니다
3. 다시 **선형성**을 이용해 개별 응답들을 합산할 수 있습니다

### 1.2 임펄스를 이용한 신호 표현

임의의 이산 시간 신호는 이동된 임펄스들의 가중합으로 표현할 수 있습니다:

$$x[n] = \sum_{k=-\infty}^{\infty} x[k] \delta[n - k]$$

이는 **추출 성질(sifting property)**을 재기술한 것입니다. 각 $x[k]\delta[n-k]$는 스케일된 이동 임펄스입니다.

연속 시간의 경우:

$$x(t) = \int_{-\infty}^{\infty} x(\tau) \delta(t - \tau) \, d\tau$$

이 분해가 합성곱을 가능하게 하는 핵심 열쇠입니다.

---

## 2. 임펄스 응답

### 2.1 정의

**임펄스 응답** $h(t)$ (또는 $h[n]$)은 입력이 단위 임펄스일 때 시스템의 출력입니다:

$$h(t) = \mathcal{T}\{\delta(t)\}, \qquad h[n] = \mathcal{T}\{\delta[n]\}$$

LTI 시스템의 경우, 임펄스 응답이 시스템을 **완전히 특성화**합니다. 그 외의 정보는 필요하지 않습니다.

### 2.2 임펄스 응답이 시스템을 특성화하는 이유

**이산 시간 유도**: 입력 $\delta[n]$이 $h[n]$을 만들어내면, 시불변성에 의해 $\delta[n-k]$는 $h[n-k]$를 만들어냅니다. 동차성(homogeneity)에 의해 $x[k]\delta[n-k]$는 $x[k]h[n-k]$를 만들어냅니다. 가산성(additivity)에 의해:

$$y[n] = \mathcal{T}\left\{\sum_k x[k]\delta[n-k]\right\} = \sum_k x[k] h[n-k]$$

이것이 **합성곱 합(convolution sum)**입니다.

**연속 시간 유도**: 마찬가지로:

$$y(t) = \mathcal{T}\left\{\int x(\tau)\delta(t-\tau)d\tau\right\} = \int x(\tau) h(t - \tau) \, d\tau$$

이것이 **합성곱 적분(convolution integral)**입니다.

### 2.3 임펄스 응답의 예

| 시스템 | 임펄스 응답 | 특성 |
|--------|-----------------|------------|
| 이상적 $D$ 샘플 지연 | $h[n] = \delta[n - D]$ | FIR, 인과적, 안정 |
| 이동 평균 (길이 $M$) | $h[n] = \frac{1}{M}\sum_{k=0}^{M-1}\delta[n-k]$ | FIR, 인과적, 안정 |
| 1차 재귀 | $h[n] = a^n u[n]$ | IIR, 인과적, $\|a\| < 1$이면 안정 |
| 이상적 저역통과 필터 | $h(t) = \text{sinc}(2Bt)$ | 비인과적, 실현 불가 |
| RC 저역통과 회로 | $h(t) = \frac{1}{RC}e^{-t/RC}u(t)$ | 인과적, 안정 |

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Impulse response examples ---
n = np.arange(-5, 30)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# 1. Ideal delay (D=3)
D = 3
h_delay = np.where(n == D, 1.0, 0.0)
axes[0, 0].stem(n, h_delay, linefmt='b-', markerfmt='bo', basefmt='k-')
axes[0, 0].set_title(f'Ideal Delay: $h[n] = \\delta[n - {D}]$')
axes[0, 0].set_xlabel('n')
axes[0, 0].set_ylabel('h[n]')
axes[0, 0].grid(True, alpha=0.3)

# 2. Moving average (M=5)
M = 5
h_ma = np.where((n >= 0) & (n < M), 1.0 / M, 0.0)
axes[0, 1].stem(n, h_ma, linefmt='r-', markerfmt='ro', basefmt='k-')
axes[0, 1].set_title(f'Moving Average (M={M}): $h[n] = \\frac{{1}}{{{M}}}$, $0 \\leq n < {M}$')
axes[0, 1].set_xlabel('n')
axes[0, 1].set_ylabel('h[n]')
axes[0, 1].grid(True, alpha=0.3)

# 3. First-order recursive (a=0.8)
a = 0.8
h_recursive = np.where(n >= 0, a**n, 0.0)
axes[1, 0].stem(n, h_recursive, linefmt='g-', markerfmt='go', basefmt='k-')
axes[1, 0].set_title(f'First-Order Recursive: $h[n] = ({a})^n u[n]$')
axes[1, 0].set_xlabel('n')
axes[1, 0].set_ylabel('h[n]')
axes[1, 0].grid(True, alpha=0.3)

# 4. RC lowpass (continuous-time, simulated)
t = np.linspace(-1, 8, 1000)
RC = 1.0
h_rc = np.where(t >= 0, (1 / RC) * np.exp(-t / RC), 0.0)
axes[1, 1].plot(t, h_rc, 'm-', linewidth=2)
axes[1, 1].set_title(f'RC Lowpass: $h(t) = \\frac{{1}}{{RC}}e^{{-t/RC}}u(t)$, RC={RC}')
axes[1, 1].set_xlabel('t')
axes[1, 1].set_ylabel('h(t)')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].fill_between(t[t >= 0], h_rc[t >= 0], alpha=0.2, color='m')

plt.tight_layout()
plt.savefig('impulse_responses.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 3. 합성곱 합 (이산 시간)

### 3.1 정의

**합성곱 합(convolution sum)**은 임펄스 응답 $h[n]$과 입력 $x[n]$을 가진 이산 시간 LTI 시스템의 출력 $y[n]$을 계산합니다:

$$y[n] = x[n] * h[n] = \sum_{k=-\infty}^{\infty} x[k] \, h[n - k]$$

별표 $*$는 합성곱을 나타냅니다 (곱셈이 아님).

### 3.2 해석

각 출력 샘플 $y[n]$에 대해:

1. $h[k]$를 $k = 0$ 기준으로 **뒤집어** $h[-k]$를 얻습니다
2. $n$만큼 **이동**하여 $h[n - k]$를 얻습니다
3. $x[k]$와 $h[n - k]$를 요소별로 **곱합니다**
4. 모든 $k$에 대해 **합산합니다**

이 "뒤집기-이동-곱하기-합산"의 절차가 합성곱을 계산하는 기계적인 방법입니다.

### 3.3 그래프를 이용한 합성곱

손으로 계산할 때는 그래프를 이용한 합성곱이 매우 유용합니다. $n$을 고정하고, $k$가 변함에 따라 $x[k]$와 $h[n-k]$의 겹침을 시각화하는 방법입니다.

**예제**: $x[n] = \{1, 2, 3\}$ ($n = 0, 1, 2$)과 $h[n] = \{1, 1, 1, 1\}$ ($n = 0, 1, 2, 3$)의 합성곱:

출력 길이는 $\text{len}(x) + \text{len}(h) - 1 = 3 + 4 - 1 = 6$입니다.

| $n$ | 겹치는 곱 | $y[n]$ |
|-----|---------------------|--------|
| 0 | $1 \cdot 1$ | 1 |
| 1 | $1 \cdot 2 + 1 \cdot 1$ | 3 |
| 2 | $1 \cdot 3 + 1 \cdot 2 + 1 \cdot 1$ | 6 |
| 3 | $1 \cdot 3 + 1 \cdot 2 + 1 \cdot 1$ | 6 |
| 4 | $1 \cdot 3 + 1 \cdot 2$ | 5 |
| 5 | $1 \cdot 3$ | 3 |

따라서 $y[n] = \{1, 3, 6, 6, 5, 3\}$.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Graphical convolution step-by-step ---
x = np.array([1, 2, 3])
h = np.array([1, 1, 1, 1])

# Full convolution
y = np.convolve(x, h)
print("x =", x)
print("h =", h)
print("y = x * h =", y)

# Visualize step-by-step
fig, axes = plt.subplots(3, 2, figsize=(14, 10))

n_values = [0, 1, 2, 3, 4, 5]
k = np.arange(-2, 8)

for idx, (ax, n_val) in enumerate(zip(axes.flat, n_values)):
    # x[k]
    x_full = np.zeros_like(k, dtype=float)
    for i, ki in enumerate(k):
        if 0 <= ki < len(x):
            x_full[i] = x[ki]

    # h[n-k] (flipped and shifted)
    h_shifted = np.zeros_like(k, dtype=float)
    for i, ki in enumerate(k):
        idx_h = n_val - ki
        if 0 <= idx_h < len(h):
            h_shifted[i] = h[idx_h]

    # Product
    product = x_full * h_shifted

    ax.stem(k, x_full, linefmt='b-', markerfmt='bo', basefmt='k-',
            label='$x[k]$')
    ax.stem(k + 0.15, h_shifted, linefmt='r-', markerfmt='rs', basefmt='k-',
            label='$h[n-k]$')

    # Highlight overlap region
    overlap_mask = product != 0
    if np.any(overlap_mask):
        for ki, pi in zip(k[overlap_mask], product[overlap_mask]):
            ax.annotate(f'{pi:.0f}', (ki + 0.07, max(x_full[k == ki][0],
                        h_shifted[k == ki][0]) + 0.15),
                        ha='center', fontsize=9, color='green', fontweight='bold')

    ax.set_title(f'$n = {n_val}$: $y[{n_val}] = {y[n_val]:.0f}$')
    ax.set_xlabel('k')
    ax.legend(fontsize=8)
    ax.set_ylim([-0.5, 4])
    ax.grid(True, alpha=0.3)

plt.suptitle('Graphical Convolution: Flip-Shift-Multiply-Sum', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('graphical_convolution.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 3.4 특수한 합성곱 결과

**임펄스와의 합성곱** (항등원):

$$x[n] * \delta[n] = x[n]$$

**이동된 임펄스와의 합성곱** (지연):

$$x[n] * \delta[n - n_0] = x[n - n_0]$$

**단위 계단과의 합성곱**:

$$x[n] * u[n] = \sum_{k=-\infty}^{n} x[k] \quad \text{(누적 합 / 누산기)}$$

---

## 4. 합성곱 적분 (연속 시간)

### 4.1 정의

**합성곱 적분(convolution integral)**은 연속 시간에서의 대응 연산입니다:

$$y(t) = x(t) * h(t) = \int_{-\infty}^{\infty} x(\tau) \, h(t - \tau) \, d\tau$$

### 4.2 해석

절차는 이산 시간의 경우를 그대로 반영합니다:

1. **뒤집기**: $h(\tau) \to h(-\tau)$
2. **이동**: $h(-\tau) \to h(t - \tau)$
3. **곱하기**: $x(\tau) \cdot h(t - \tau)$
4. **적분**: $\int_{-\infty}^{\infty} (\cdot) \, d\tau$

### 4.3 예제: 지수 함수와 계단 함수의 합성곱

$a > 0$일 때 $y(t) = e^{-at}u(t) * u(t)$를 계산합니다.

$$y(t) = \int_{-\infty}^{\infty} e^{-a\tau}u(\tau) \cdot u(t - \tau) \, d\tau$$

피적분 함수는 $u(\tau)$에 의해 $\tau \geq 0$이고, $u(t-\tau)$에 의해 $\tau \leq t$일 때만 0이 아니므로, $t \geq 0$에서:

$$y(t) = \int_0^t e^{-a\tau} \, d\tau = \frac{1}{a}(1 - e^{-at}), \quad t \geq 0$$

$t < 0$에서는: $y(t) = 0$.

따라서: $y(t) = \frac{1}{a}(1 - e^{-at})u(t)$

### 4.4 예제: 직사각형 펄스의 자기 합성곱

$\text{rect}(t)$가 $|t| \leq 1/2$에서 1인 경우, $y(t) = \text{rect}(t) * \text{rect}(t)$를 계산합니다.

뒤집기-이동-적분 절차에 의해:

$$y(t) = \int_{-\infty}^{\infty} \text{rect}(\tau) \cdot \text{rect}(t - \tau) \, d\tau$$

0과 $t$를 중심으로 하는 너비 1의 두 직사각형의 겹침은 **삼각형 펄스**를 만들어냅니다:

$$y(t) = \text{tri}(t) = \begin{cases} 1 - |t| & |t| \leq 1 \\ 0 & |t| > 1 \end{cases}$$

> 이것은 기본적인 결과입니다: 직사각형과 자신의 합성곱은 삼각형을 만들어냅니다. 한 번 더 합성곱하면 더 부드러운 형태가 되고, 극한에서는 (중심 극한 정리에 의해) 가우시안에 수렴합니다.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Continuous-time convolution examples ---

# Example 1: exp(-at)u(t) * u(t)
a = 2.0
t = np.linspace(-1, 5, 1000)
dt = t[1] - t[0]

h = np.where(t >= 0, np.exp(-a * t), 0.0)
x = np.where(t >= 0, 1.0, 0.0)

# Numerical convolution
y_numerical = np.convolve(h, x, mode='full') * dt
t_conv = np.arange(len(y_numerical)) * dt + 2 * t[0]

# Analytical result
y_analytical = np.where(t >= 0, (1/a) * (1 - np.exp(-a * t)), 0.0)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

axes[0, 0].plot(t, h, 'b-', linewidth=2)
axes[0, 0].set_title('$h(t) = e^{-2t}u(t)$')
axes[0, 0].set_xlabel('t')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(t, x, 'r-', linewidth=2)
axes[0, 1].set_title('$x(t) = u(t)$')
axes[0, 1].set_xlabel('t')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(t, y_analytical, 'g-', linewidth=2, label='Analytical')
axes[1, 0].plot(t_conv[:len(t)], y_numerical[:len(t)], 'k--', linewidth=1.5,
                label='Numerical', alpha=0.7)
axes[1, 0].set_title('$y(t) = \\frac{1}{a}(1 - e^{-at})u(t)$')
axes[1, 0].set_xlabel('t')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Example 2: rect * rect = triangle
t2 = np.linspace(-2, 2, 1000)
dt2 = t2[1] - t2[0]
rect = np.where(np.abs(t2) <= 0.5, 1.0, 0.0)
tri_conv = np.convolve(rect, rect, mode='full') * dt2
t2_conv = np.arange(len(tri_conv)) * dt2 + 2 * t2[0]

tri_analytical = np.maximum(0, 1 - np.abs(t2))

axes[1, 1].plot(t2, tri_analytical, 'purple', linewidth=2, label='tri(t) analytical')
axes[1, 1].plot(t2_conv[:len(t2)] + t2[0], tri_conv[:len(t2)], 'k--',
                linewidth=1.5, label='rect*rect numerical', alpha=0.7)
axes[1, 1].set_title('rect$(t)$ * rect$(t)$ = tri$(t)$')
axes[1, 1].set_xlabel('t')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('convolution_examples.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 4.5 인과 신호와의 합성곱

$x(t)$와 $h(t)$ 모두 인과적(causal)일 때(즉, $t < 0$에서 0), 적분 한계가 단순해집니다:

$$y(t) = \int_0^t x(\tau) h(t - \tau) \, d\tau, \quad t \geq 0$$

이는 $\tau < 0$에서 $x(\tau) = 0$이고, $\tau > t$에서 $h(t - \tau) = 0$이기 때문입니다.

---

## 5. 합성곱 계산

### 5.1 해석적 방법

**방법 1: 직접 적분** — 피적분 함수가 0이 아닌 적분 한계를 파악한 후 적분합니다.

**방법 2: 라플라스/Z 변환** — 변환 영역에서 합성곱은 곱셈이 됩니다:

$$\mathcal{L}\{x * h\} = X(s) \cdot H(s)$$

$$\mathcal{Z}\{x * h\} = X(z) \cdot H(z)$$

이 방법은 직접 계산보다 훨씬 단순한 경우가 많습니다. 이후 레슨에서 자세히 다룰 것입니다.

### 5.2 수치적 방법

**직접 구현** (단순 방법, $O(N^2)$):

```python
def convolve_direct(x, h):
    """Direct convolution: O(N*M) where N=len(x), M=len(h)."""
    N = len(x)
    M = len(h)
    y = np.zeros(N + M - 1)
    for n in range(N + M - 1):
        for k in range(M):
            if 0 <= n - k < N:
                y[n] += h[k] * x[n - k]
    return y
```

**FFT 기반 합성곱** (빠른 방법, $O(N \log N)$):

```python
def convolve_fft(x, h):
    """FFT-based convolution: O(N log N)."""
    N = len(x) + len(h) - 1
    # Pad to next power of 2 for FFT efficiency
    N_fft = 2 ** int(np.ceil(np.log2(N)))
    X = np.fft.fft(x, N_fft)
    H = np.fft.fft(h, N_fft)
    y = np.real(np.fft.ifft(X * H))
    return y[:N]
```

FFT 방법은 **합성곱 정리(convolution theorem)**를 활용합니다: 시간 영역에서의 합성곱은 주파수 영역에서의 곱셈과 같습니다. 큰 신호에 대해 이 방법은 극적으로 더 빠릅니다.

### 5.3 NumPy 및 SciPy 함수

```python
import numpy as np
from scipy import signal

x = np.array([1, 2, 3, 4, 5])
h = np.array([0.2, 0.3, 0.5])

# Full convolution (output length = len(x) + len(h) - 1)
y_full = np.convolve(x, h, mode='full')
print("Full:", y_full)

# Same-size output (centered, length = max(len(x), len(h)))
y_same = np.convolve(x, h, mode='same')
print("Same:", y_same)

# Valid (only where signals fully overlap, length = max(N,M) - min(N,M) + 1)
y_valid = np.convolve(x, h, mode='valid')
print("Valid:", y_valid)

# scipy.signal.fftconvolve for large arrays (FFT-based)
y_fft = signal.fftconvolve(x, h, mode='full')
print("FFT:", y_fft)
```

### 5.4 합성곱 모드 비교

| 모드 | 출력 길이 | 설명 |
|------|-------------|-------------|
| `'full'` | $N + M - 1$ | 완전한 합성곱 결과 |
| `'same'` | $\max(N, M)$ | 가장 큰 입력과 동일한 크기의 출력 |
| `'valid'` | $|N - M| + 1$ | 입력들이 완전히 겹치는 부분만 |

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Convolution modes visualization ---
x = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
h = np.array([0.2, 0.6, 0.2])

y_full = np.convolve(x, h, mode='full')
y_same = np.convolve(x, h, mode='same')
y_valid = np.convolve(x, h, mode='valid')

fig, axes = plt.subplots(4, 1, figsize=(12, 10))

axes[0].stem(range(len(x)), x, linefmt='b-', markerfmt='bo', basefmt='k-')
axes[0].set_title(f'Input x[n] (length {len(x)})')
axes[0].grid(True, alpha=0.3)

axes[1].stem(range(len(y_full)), y_full, linefmt='r-', markerfmt='ro', basefmt='k-')
axes[1].set_title(f"mode='full' (length {len(y_full)} = {len(x)} + {len(h)} - 1)")
axes[1].grid(True, alpha=0.3)

axes[2].stem(range(len(y_same)), y_same, linefmt='g-', markerfmt='go', basefmt='k-')
axes[2].set_title(f"mode='same' (length {len(y_same)})")
axes[2].grid(True, alpha=0.3)

axes[3].stem(range(len(y_valid)), y_valid, linefmt='m-', markerfmt='mo', basefmt='k-')
axes[3].set_title(f"mode='valid' (length {len(y_valid)})")
axes[3].grid(True, alpha=0.3)

for ax in axes:
    ax.set_xlabel('n')

plt.tight_layout()
plt.savefig('convolution_modes.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 6. 합성곱의 성질

합성곱은 이론적으로 중요하고 실용적으로도 유용한 몇 가지 대수적 성질을 만족합니다.

### 6.1 교환법칙(Commutativity)

$$x[n] * h[n] = h[n] * x[n]$$

동치 표현:

$$\sum_k x[k]h[n-k] = \sum_k h[k]x[n-k]$$

**증명**: 첫 번째 합에서 $m = n - k$로 치환합니다.

**의미**: 입력과 시스템의 역할을 교환할 수 있습니다. 합성곱 계산 시, 어느 신호를 "뒤집을지" 편의에 따라 선택할 수 있습니다.

### 6.2 결합법칙(Associativity)

$$(x * h_1) * h_2 = x * (h_1 * h_2)$$

**의미**: 임펄스 응답 $h_1$과 $h_2$를 가진 두 LTI 시스템의 직렬 연결은, 직렬 연결 순서에 무관하게, 임펄스 응답 $h_1 * h_2$를 가진 단일 LTI 시스템과 동등합니다.

### 6.3 덧셈에 대한 분배법칙(Distributivity over Addition)

$$x * (h_1 + h_2) = x * h_1 + x * h_2$$

**의미**: 두 LTI 시스템의 병렬 조합은 개별 임펄스 응답의 합을 임펄스 응답으로 갖는 단일 시스템과 동등합니다.

### 6.4 항등원(Identity Element)

$$x * \delta = x$$

임펄스 $\delta$는 합성곱의 항등원입니다. 곱셈에서 1이 항등원인 것과 같습니다.

### 6.5 이동 성질(Shift Property)

$$x[n] * \delta[n - n_0] = x[n - n_0]$$

이동된 임펄스와의 합성곱은 신호를 지연시킵니다.

### 6.6 폭 성질(Width Property)

$x[n]$의 지지(support)가 $[N_1, N_2]$이고, $h[n]$의 지지가 $[M_1, M_2]$이면, $y[n] = x[n] * h[n]$의 지지는 $[N_1 + M_1, N_2 + M_2]$입니다.

출력의 **지속 시간**(폭)은 입력 지속 시간의 합과 같습니다(이산 시간에서는 1을 뺍니다).

### 6.7 스케일된 임펄스 쌍과의 합성곱 (에코 시스템)

일반적인 응용: 시스템 $h[n] = \delta[n] + \alpha \delta[n - D]$ (지연 $D$와 감쇠 $\alpha$를 가진 에코)는 다음을 생성합니다:

$$y[n] = x[n] + \alpha x[n - D]$$

```python
import numpy as np

# --- Verify convolution properties ---
np.random.seed(42)
x = np.random.randn(20)
h1 = np.random.randn(10)
h2 = np.random.randn(8)

# Commutativity
y1 = np.convolve(x, h1)
y2 = np.convolve(h1, x)
print(f"Commutativity error: {np.max(np.abs(y1 - y2)):.2e}")

# Associativity
y_assoc1 = np.convolve(np.convolve(x, h1), h2)
y_assoc2 = np.convolve(x, np.convolve(h1, h2))
print(f"Associativity error: {np.max(np.abs(y_assoc1 - y_assoc2)):.2e}")

# Distributivity
y_dist1 = np.convolve(x, h1 + h2[:len(h1)])  # need same length for addition
h_padded = np.zeros(max(len(h1), len(h2)))
h_padded[:len(h1)] += h1
h_padded2 = np.zeros(max(len(h1), len(h2)))
h_padded2[:len(h2)] += h2
y_dist_lhs = np.convolve(x, h_padded + h_padded2)
y_dist_rhs = np.zeros(len(y_dist_lhs))
y_r1 = np.convolve(x, h_padded)
y_r2 = np.convolve(x, h_padded2)
max_len = max(len(y_r1), len(y_r2))
y_dist_rhs_a = np.zeros(max_len)
y_dist_rhs_a[:len(y_r1)] += y_r1
y_dist_rhs_a[:len(y_r2)] += y_r2
print(f"Distributivity error: {np.max(np.abs(y_dist_lhs - y_dist_rhs_a[:len(y_dist_lhs)])):.2e}")

# Identity
delta = np.zeros(1)
delta[0] = 1.0
y_id = np.convolve(x, delta)
print(f"Identity error: {np.max(np.abs(x - y_id[:len(x)])):.2e}")
```

---

## 7. 계단 응답

### 7.1 정의

**계단 응답(step response)** $s(t)$ (또는 $s[n]$)은 입력이 단위 계단일 때의 출력입니다:

$$s(t) = h(t) * u(t) = \int_{-\infty}^{t} h(\tau) \, d\tau$$

$$s[n] = h[n] * u[n] = \sum_{k=-\infty}^{n} h[k]$$

### 7.2 임펄스 응답과의 관계

계단 응답은 임펄스 응답의 **누적 적분**(연속 시간) 또는 **누적 합**(이산 시간)입니다:

$$s(t) = \int_{-\infty}^{t} h(\tau) \, d\tau \quad \Leftrightarrow \quad h(t) = \frac{ds(t)}{dt}$$

$$s[n] = \sum_{k=-\infty}^{n} h[k] \quad \Leftrightarrow \quad h[n] = s[n] - s[n-1]$$

즉, 계단 응답을 미분하거나(연속) 1차 차분을 취해(이산) 임펄스 응답을 구할 수 있으며, 반대도 가능합니다.

### 7.3 예제

$h[n] = (0.8)^n u[n]$에 대해:

$$s[n] = \sum_{k=0}^{n} (0.8)^k = \frac{1 - (0.8)^{n+1}}{1 - 0.8} = 5(1 - (0.8)^{n+1}), \quad n \geq 0$$

$n \to \infty$에서: $s[\infty] = 5$, 이것이 시스템의 **DC 이득**입니다.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Impulse response and step response ---
n = np.arange(0, 30)
a = 0.8

h = a ** n  # impulse response
s = np.cumsum(h)  # step response = running sum of h

# Analytical step response
s_analytical = 5 * (1 - 0.8 ** (n + 1))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].stem(n, h, linefmt='b-', markerfmt='bo', basefmt='k-')
axes[0].set_title(f'Impulse Response: $h[n] = ({a})^n u[n]$')
axes[0].set_xlabel('n')
axes[0].set_ylabel('h[n]')
axes[0].grid(True, alpha=0.3)

axes[1].stem(n, s, linefmt='r-', markerfmt='ro', basefmt='k-', label='Numerical')
axes[1].plot(n, s_analytical, 'k--', linewidth=1.5, label='Analytical')
axes[1].axhline(y=5, color='gray', linestyle=':', label='DC gain = 5')
axes[1].set_title(f'Step Response: $s[n] = 5(1 - {a}^{{n+1}})$')
axes[1].set_xlabel('n')
axes[1].set_ylabel('s[n]')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('step_response.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 8. 시스템 상호연결

LTI 시스템은 여러 방식으로 결합할 수 있으며, 결과로 얻는 합성 시스템도 LTI입니다.

### 8.1 직렬(Cascade) 연결

```
x → [h₁] → [h₂] → y
```

**등가 임펄스 응답**: $h_{\text{eq}} = h_1 * h_2$

합성곱의 결합법칙과 교환법칙에 의해:

$$y = x * h_1 * h_2 = x * (h_1 * h_2) = x * (h_2 * h_1)$$

직렬 연결의 순서는 (LTI 시스템에서) 중요하지 않습니다.

### 8.2 병렬(Parallel) 연결

```
     ┌─[h₁]─┐
x ──►│       ├──► (+) → y
     └─[h₂]─┘
```

**등가 임펄스 응답**: $h_{\text{eq}} = h_1 + h_2$

분배법칙에 의해:

$$y = x * h_1 + x * h_2 = x * (h_1 + h_2)$$

### 8.3 피드백(Feedback) 연결

```
x → (+) → [h₁] → y
      ↑            │
      └── [h₂] ◄──┘
```

음의 피드백: $e[n] = x[n] - h_2[n] * y[n]$이고 $y[n] = h_1[n] * e[n]$.

이를 정리하면: $y = h_1 * (x - h_2 * y)$

변환 영역(합성곱이 곱셈이 되는)에서:

$$Y(z) = H_1(z)(X(z) - H_2(z)Y(z))$$

$$H_{\text{eq}}(z) = \frac{Y(z)}{X(z)} = \frac{H_1(z)}{1 + H_1(z)H_2(z)}$$

피드백은 IIR(재귀) 필터와 제어 시스템을 구현하는 데 필수적입니다.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- System interconnections ---
n = np.arange(0, 40)

# Two simple systems
a1, a2 = 0.7, 0.5
h1 = np.where(n >= 0, a1**n, 0.0)
h2 = np.where(n >= 0, a2**n, 0.0)

# Input signal
x = np.zeros(40)
x[0] = 1.0  # impulse input

# Cascade
h_cascade = np.convolve(h1, h2)[:40]
y_cascade = np.convolve(x, h_cascade)[:40]

# Parallel
h_parallel = h1 + h2
y_parallel = np.convolve(x, h_parallel)[:40]

# Verify cascade commutativity
h_cascade_rev = np.convolve(h2, h1)[:40]

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

axes[0, 0].stem(n, h1, linefmt='b-', markerfmt='bo', basefmt='k-', label='$h_1$')
axes[0, 0].stem(n + 0.2, h2, linefmt='r-', markerfmt='rs', basefmt='k-', label='$h_2$')
axes[0, 0].set_title('Individual Impulse Responses')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].stem(n, h_cascade, linefmt='g-', markerfmt='go', basefmt='k-',
                label='$h_1 * h_2$')
axes[0, 1].stem(n + 0.2, h_cascade_rev, linefmt='m-', markerfmt='ms', basefmt='k-',
                label='$h_2 * h_1$', alpha=0.5)
axes[0, 1].set_title('Cascade: $h_1 * h_2 = h_2 * h_1$')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].stem(n, h_parallel, linefmt='orange', markerfmt='o', basefmt='k-')
axes[1, 0].set_title('Parallel: $h_1 + h_2$')
axes[1, 0].grid(True, alpha=0.3)

# Compare cascade and parallel responses
axes[1, 1].stem(n, y_cascade, linefmt='g-', markerfmt='go', basefmt='k-',
                label='Cascade')
axes[1, 1].stem(n + 0.2, y_parallel, linefmt='orange', markerfmt='o', basefmt='k-',
                label='Parallel', alpha=0.7)
axes[1, 1].set_title('Impulse Response Comparison')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

for ax in axes.flat:
    ax.set_xlabel('n')

plt.tight_layout()
plt.savefig('system_interconnections.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 9. LTI 시스템 안정성

### 9.1 BIBO 안정성 판별 기준

레슨 01에서 확립했듯이, LTI 시스템은 임펄스 응답이 **절대 합산 가능**(이산) 또는 **절대 적분 가능**(연속)한 경우에만 BIBO 안정합니다:

**이산 시간**:

$$\sum_{n=-\infty}^{\infty} |h[n]| < \infty$$

**연속 시간**:

$$\int_{-\infty}^{\infty} |h(t)| \, dt < \infty$$

### 9.2 일반적인 시스템의 안정성

**1차 재귀**: $h[n] = a^n u[n]$

$$\sum_{n=0}^{\infty} |a|^n = \frac{1}{1 - |a|} < \infty \iff |a| < 1$$

극점이 단위원 내에 있을 때 안정합니다.

**FIR 시스템**: $h[n]$의 지속 시간이 유한한 경우 (예: $N$ 샘플)

$$\sum_{n} |h[n]| = \text{유한한 합} < \infty$$

FIR 시스템은 (계수 값이 유한하다면) **항상 BIBO 안정**합니다.

**이상적 적분기 / 누산기**: $h[n] = u[n]$

$$\sum_{n=0}^{\infty} 1 = \infty$$

불안정합니다. 상수 입력은 선형적으로 증가하는 출력을 만들어냅니다.

### 9.3 인과성과 안정성

인과적 LTI 시스템의 경우, 임펄스 응답은 $n < 0$ (또는 $t < 0$)에서 $h[n] = 0$ (또는 $h(t) = 0$)을 만족합니다.

Z 변환 영역에서, 인과적 시스템은 $H(z)$의 모든 극점이 단위원 **내부** $|z| < 1$에 있을 때 안정합니다.

라플라스 영역에서, 인과적 시스템은 $H(s)$의 모든 극점이 **좌반평면** $\text{Re}(s) < 0$에 있을 때 안정합니다.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Stability analysis for first-order systems ---
n = np.arange(0, 50)

poles = [0.3, 0.7, 0.95, 1.0, 1.05]
labels = ['a=0.3 (stable)', 'a=0.7 (stable)', 'a=0.95 (stable)',
          'a=1.0 (marginally unstable)', 'a=1.05 (unstable)']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for a, label in zip(poles, labels):
    h = a ** n
    axes[0].plot(n, h, linewidth=1.5, label=label)

axes[0].set_title('Impulse Response $h[n] = a^n u[n]$')
axes[0].set_xlabel('n')
axes[0].set_ylabel('h[n]')
axes[0].set_ylim([-1, 10])
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Cumulative sum of |h[n]| (should converge for stable systems)
for a, label in zip(poles, labels):
    h = a ** n
    cum_sum = np.cumsum(np.abs(h))
    axes[1].plot(n, cum_sum, linewidth=1.5, label=label)

axes[1].set_title('Running Sum $\\sum_{k=0}^{n} |h[k]|$')
axes[1].set_xlabel('n')
axes[1].set_ylabel('Partial sum')
axes[1].set_ylim([0, 50])
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('stability_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 10. LTI 시스템의 주파수 응답

### 10.1 복소 지수 함수를 고유 함수로

주파수 영역 분석을 가능하게 하는 핵심 성질:

> LTI 시스템의 입력이 복소 지수 $x[n] = e^{j\omega n}$이면, 출력은:
>
> $$y[n] = H(e^{j\omega}) \cdot e^{j\omega n}$$
>
> 이며, $H(e^{j\omega})$는 시스템의 **주파수 응답(frequency response)**입니다.

**증명**:

$$y[n] = \sum_k h[k] e^{j\omega(n-k)} = e^{j\omega n} \sum_k h[k] e^{-j\omega k} = e^{j\omega n} H(e^{j\omega})$$

여기서:

$$H(e^{j\omega}) = \sum_{n=-\infty}^{\infty} h[n] e^{-j\omega n}$$

이것이 임펄스 응답의 **이산 시간 푸리에 변환(DTFT, Discrete-Time Fourier Transform)**입니다.

### 10.2 크기 응답과 위상 응답

주파수 응답은 일반적으로 복소수입니다:

$$H(e^{j\omega}) = |H(e^{j\omega})| e^{j\angle H(e^{j\omega})}$$

- $|H(e^{j\omega})|$은 **크기 응답(magnitude response)** — 시스템이 각 주파수를 얼마나 증폭 또는 감쇠시키는가
- $\angle H(e^{j\omega})$은 **위상 응답(phase response)** — 시스템이 각 주파수 성분을 얼마나 지연시키는가

### 10.3 연속 시간 주파수 응답

연속 시간 LTI 시스템의 경우:

$$H(j\omega) = \int_{-\infty}^{\infty} h(t) e^{-j\omega t} \, dt$$

이것은 $h(t)$의 **푸리에 변환(Fourier Transform)**입니다. 입력이 $x(t) = e^{j\omega_0 t}$이면, 출력은 $y(t) = H(j\omega_0)e^{j\omega_0 t}$입니다.

### 10.4 예제: 이동 평균 필터

5점 이동 평균 필터: $n = 0, 1, 2, 3, 4$에서 $h[n] = \frac{1}{5}$:

$$H(e^{j\omega}) = \frac{1}{5} \sum_{n=0}^{4} e^{-j\omega n} = \frac{1}{5} \cdot \frac{1 - e^{-j5\omega}}{1 - e^{-j\omega}}$$

크기:

$$|H(e^{j\omega})| = \frac{1}{5} \left|\frac{\sin(5\omega/2)}{\sin(\omega/2)}\right|$$

이것은 저역통과 필터입니다 — 저주파는 통과시키고 고주파는 감쇠시킵니다.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Frequency response of common systems ---
omega = np.linspace(-np.pi, np.pi, 1000)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Moving average (M=5)
M = 5
H_ma = np.zeros_like(omega, dtype=complex)
for n in range(M):
    H_ma += (1 / M) * np.exp(-1j * omega * n)

axes[0, 0].plot(omega / np.pi, 20 * np.log10(np.abs(H_ma) + 1e-12), 'b-', linewidth=2)
axes[0, 0].set_title(f'Moving Average (M={M}) — Magnitude Response')
axes[0, 0].set_xlabel('Normalized Frequency ($\\omega/\\pi$)')
axes[0, 0].set_ylabel('Magnitude (dB)')
axes[0, 0].set_ylim([-30, 5])
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(omega / np.pi, np.angle(H_ma), 'r-', linewidth=2)
axes[0, 1].set_title(f'Moving Average (M={M}) — Phase Response')
axes[0, 1].set_xlabel('Normalized Frequency ($\\omega/\\pi$)')
axes[0, 1].set_ylabel('Phase (radians)')
axes[0, 1].grid(True, alpha=0.3)

# 2. First-order IIR: h[n] = a^n u[n], H(z) = 1/(1-az^{-1})
for a in [0.5, 0.8, 0.95]:
    H_iir = 1 / (1 - a * np.exp(-1j * omega))
    axes[1, 0].plot(omega / np.pi, 20 * np.log10(np.abs(H_iir)),
                    linewidth=2, label=f'a={a}')

axes[1, 0].set_title('First-Order IIR — Magnitude Response')
axes[1, 0].set_xlabel('Normalized Frequency ($\\omega/\\pi$)')
axes[1, 0].set_ylabel('Magnitude (dB)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 3. Effect of filtering on a signal
np.random.seed(42)
N = 500
n_sig = np.arange(N)
x_signal = np.sin(2 * np.pi * 0.05 * n_sig) + 0.5 * np.sin(2 * np.pi * 0.4 * n_sig)
x_noisy = x_signal + 0.3 * np.random.randn(N)

# Apply 11-point moving average
M_filt = 11
h_filt = np.ones(M_filt) / M_filt
y_filtered = np.convolve(x_noisy, h_filt, mode='same')

axes[1, 1].plot(n_sig, x_noisy, 'gray', linewidth=0.5, alpha=0.7, label='Noisy input')
axes[1, 1].plot(n_sig, y_filtered, 'b-', linewidth=1.5, label=f'MA({M_filt}) filtered')
axes[1, 1].plot(n_sig, np.sin(2 * np.pi * 0.05 * n_sig), 'r--', linewidth=1.5,
                label='Low-freq component')
axes[1, 1].set_title('Moving Average as Lowpass Filter')
axes[1, 1].set_xlabel('n')
axes[1, 1].legend(fontsize=9)
axes[1, 1].set_xlim([0, 200])
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('frequency_response.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 11. Python 예제

### 11.1 완전한 합성곱 툴킷

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time

class ConvolutionToolkit:
    """Comprehensive toolkit for computing and analyzing convolutions."""

    @staticmethod
    def convolve_direct(x, h):
        """Direct O(NM) convolution."""
        N, M = len(x), len(h)
        y = np.zeros(N + M - 1)
        for n in range(N + M - 1):
            for k in range(M):
                if 0 <= n - k < N:
                    y[n] += h[k] * x[n - k]
        return y

    @staticmethod
    def convolve_fft(x, h):
        """FFT-based O(N log N) convolution."""
        N = len(x) + len(h) - 1
        N_fft = 2 ** int(np.ceil(np.log2(N)))
        X = np.fft.fft(x, N_fft)
        H = np.fft.fft(h, N_fft)
        y = np.real(np.fft.ifft(X * H))
        return y[:N]

    @staticmethod
    def benchmark(N_values, M=50):
        """Compare direct vs FFT convolution speed."""
        print(f"{'N':>8} | {'Direct (ms)':>12} | {'FFT (ms)':>12} | {'NumPy (ms)':>12} | {'Speedup':>8}")
        print("-" * 60)

        for N in N_values:
            x = np.random.randn(N)
            h = np.random.randn(M)

            # Direct (only for small N)
            if N <= 5000:
                t0 = time.perf_counter()
                y_direct = ConvolutionToolkit.convolve_direct(x, h)
                t_direct = (time.perf_counter() - t0) * 1000
            else:
                t_direct = float('inf')

            # FFT
            t0 = time.perf_counter()
            y_fft = ConvolutionToolkit.convolve_fft(x, h)
            t_fft = (time.perf_counter() - t0) * 1000

            # NumPy
            t0 = time.perf_counter()
            y_np = np.convolve(x, h)
            t_numpy = (time.perf_counter() - t0) * 1000

            speedup = t_direct / t_fft if t_direct != float('inf') else float('inf')
            t_direct_str = f"{t_direct:.2f}" if t_direct != float('inf') else "skipped"
            print(f"{N:>8} | {t_direct_str:>12} | {t_fft:>12.2f} | {t_numpy:>12.2f} | {speedup:>8.1f}x")


# Run benchmark
print("=== Convolution Performance Benchmark ===\n")
ConvolutionToolkit.benchmark([100, 500, 1000, 5000, 10000, 50000])
```

### 11.2 에코 시뮬레이션

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Echo system simulation ---
fs = 8000  # sampling rate
duration = 0.5
n = np.arange(int(fs * duration))
t = n / fs

# Original signal: sum of two sinusoids
f1, f2 = 200, 500
x = 0.7 * np.sin(2 * np.pi * f1 * t) + 0.3 * np.sin(2 * np.pi * f2 * t)

# Fade in/out
fade_len = int(0.02 * fs)
x[:fade_len] *= np.linspace(0, 1, fade_len)
x[-fade_len:] *= np.linspace(1, 0, fade_len)

# Echo system: h[n] = delta[n] + 0.6*delta[n-D1] + 0.3*delta[n-D2]
D1 = int(0.1 * fs)   # 100ms delay
D2 = int(0.25 * fs)  # 250ms delay
h_echo = np.zeros(D2 + 1)
h_echo[0] = 1.0
h_echo[D1] = 0.6
h_echo[D2] = 0.3

# Apply echo
y = np.convolve(x, h_echo)
t_out = np.arange(len(y)) / fs

fig, axes = plt.subplots(3, 1, figsize=(14, 8))

axes[0].plot(t * 1000, x, 'b-', linewidth=0.8)
axes[0].set_title('Original Signal')
axes[0].set_xlabel('Time (ms)')
axes[0].set_ylabel('Amplitude')
axes[0].grid(True, alpha=0.3)

axes[1].stem(np.arange(len(h_echo)) / fs * 1000, h_echo,
             linefmt='r-', markerfmt='ro', basefmt='k-')
axes[1].set_title(f'Echo Impulse Response: $\\delta[n] + 0.6\\delta[n-{D1}] + 0.3\\delta[n-{D2}]$')
axes[1].set_xlabel('Time (ms)')
axes[1].set_ylabel('h[n]')
axes[1].grid(True, alpha=0.3)

axes[2].plot(t_out * 1000, y, 'g-', linewidth=0.8)
axes[2].set_title('Output with Echo')
axes[2].set_xlabel('Time (ms)')
axes[2].set_ylabel('Amplitude')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('echo_simulation.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 11.3 계단 응답으로부터의 시스템 식별

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Recover impulse response from step response ---

# Unknown system: 3-tap FIR h = [0.5, 1.0, 0.5]
h_true = np.array([0.5, 1.0, 0.5])

# Apply unit step to get step response
N = 30
u = np.ones(N)  # unit step
s = np.convolve(u, h_true)[:N]

# Recover impulse response by first differencing
h_recovered = np.zeros(N)
h_recovered[0] = s[0]
h_recovered[1:] = np.diff(s)

print("True impulse response:", h_true)
print("Recovered (first 5):", h_recovered[:5])

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

n = np.arange(N)

axes[0].stem(np.arange(len(h_true)), h_true, linefmt='b-', markerfmt='bo', basefmt='k-')
axes[0].set_title('True Impulse Response $h[n]$')
axes[0].set_xlabel('n')
axes[0].grid(True, alpha=0.3)

axes[1].stem(n, s, linefmt='r-', markerfmt='ro', basefmt='k-')
axes[1].set_title('Measured Step Response $s[n]$')
axes[1].set_xlabel('n')
axes[1].grid(True, alpha=0.3)

axes[2].stem(n[:10], h_recovered[:10], linefmt='g-', markerfmt='go', basefmt='k-')
axes[2].set_title('Recovered $h[n] = s[n] - s[n-1]$')
axes[2].set_xlabel('n')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('system_identification.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 11.4 주파수 응답 분석

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Frequency response from impulse response ---

def plot_freq_response(h, fs=1.0, title="System"):
    """Plot magnitude and phase response of a discrete-time system."""
    # Compute DTFT at 1024 frequency points
    N_freq = 1024
    omega = np.linspace(0, np.pi, N_freq)
    H = np.zeros(N_freq, dtype=complex)
    for k, w in enumerate(omega):
        for n, hn in enumerate(h):
            H[k] += hn * np.exp(-1j * w * n)

    mag_db = 20 * np.log10(np.abs(H) + 1e-12)
    phase = np.unwrap(np.angle(H))
    group_delay = -np.diff(phase) / np.diff(omega)

    freq = omega * fs / (2 * np.pi) if fs != 1.0 else omega / np.pi

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    # Magnitude
    axes[0].plot(freq, mag_db, 'b-', linewidth=2)
    axes[0].set_title(f'{title} — Magnitude Response')
    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].grid(True, alpha=0.3)

    # Phase
    axes[1].plot(freq, phase * 180 / np.pi, 'r-', linewidth=2)
    axes[1].set_title(f'{title} — Phase Response')
    axes[1].set_ylabel('Phase (degrees)')
    axes[1].grid(True, alpha=0.3)

    # Group delay
    axes[2].plot(freq[:-1], group_delay, 'g-', linewidth=2)
    axes[2].set_title(f'{title} — Group Delay')
    axes[2].set_xlabel('Frequency (Hz)' if fs != 1.0 else 'Normalized Frequency ($\\omega/\\pi$)')
    axes[2].set_ylabel('Samples')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'freq_response_{title.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()


# Example 1: 7-point moving average
M = 7
h_ma = np.ones(M) / M
plot_freq_response(h_ma, title="7-Point Moving Average")

# Example 2: Difference filter (high-pass)
h_diff = np.array([1, -1])
plot_freq_response(h_diff, title="First Difference")

# Example 3: Bandpass-like filter
h_bp = np.array([0.1, -0.2, 0.5, 1.0, 0.5, -0.2, 0.1])
plot_freq_response(h_bp, title="Bandpass-Like FIR")
```

---

## 12. 요약

### 핵심 공식

| 개념 | 이산 시간 | 연속 시간 |
|---------|--------------|-----------------|
| 합성곱 | $y[n] = \sum_k x[k]h[n-k]$ | $y(t) = \int x(\tau)h(t-\tau)d\tau$ |
| 임펄스 응답 | $h[n] = \mathcal{T}\{\delta[n]\}$ | $h(t) = \mathcal{T}\{\delta(t)\}$ |
| 계단 응답 | $s[n] = \sum_{k \leq n} h[k]$ | $s(t) = \int_{-\infty}^{t} h(\tau)d\tau$ |
| 주파수 응답 | $H(e^{j\omega}) = \sum_n h[n]e^{-j\omega n}$ | $H(j\omega) = \int h(t)e^{-j\omega t}dt$ |
| BIBO 안정성 | $\sum |h[n]| < \infty$ | $\int |h(t)|dt < \infty$ |

### 합성곱의 성질

| 성질 | 표현 |
|----------|-----------|
| 교환법칙 | $x * h = h * x$ |
| 결합법칙 | $(x * h_1) * h_2 = x * (h_1 * h_2)$ |
| 분배법칙 | $x * (h_1 + h_2) = x * h_1 + x * h_2$ |
| 항등원 | $x * \delta = x$ |
| 이동 | $x * \delta_{n_0} = x[n - n_0]$ |
| 폭 | $y$의 지지 = 각 지지의 합 |

### 개념 지도

```
          Impulse δ[n]
              │
              ▼
    LTI System T{·}  ──────►  Impulse Response h[n]
              │                        │
              │                ┌───────┼───────┐
              │                ▼       ▼       ▼
              │           Stability  Freq.   Step
              │           Check      Response Response
              │
    Any Input x[n]
              │
              ▼
    y[n] = x[n] * h[n]  (CONVOLUTION)
```

이 레슨의 핵심 통찰: **LTI 시스템에서 모든 것은 합성곱으로 귀결됩니다**. 임펄스 응답을 알면, 시스템에 대한 모든 것을 알 수 있습니다.

---

## 13. 연습 문제

### 연습 문제 1: 해석적 합성곱

다음 합성곱을 손으로 계산하세요. `np.convolve()`를 사용하여 결과를 검증하세요.

1. $x[n] = \{2, 1, -1\}$과 $h[n] = \{1, 3, 2\}$ ($n = 0$부터 시작)
2. $x[n] = u[n] - u[n-4]$와 $h[n] = (0.5)^n u[n]$
3. $x[n] = (0.8)^n u[n]$과 $h[n] = (0.6)^n u[n]$ (등비수열의 닫힌 형태 사용)

### 연습 문제 2: 연속 시간 합성곱

다음을 해석적으로 계산하세요:

1. $y(t) = e^{-t}u(t) * e^{-2t}u(t)$
2. $y(t) = u(t) * u(t)$ (결과는 무엇이고, 이것이 안정성 문제를 어떻게 드러내는가?)
3. $y(t) = \text{rect}(t/2) * e^{-t}u(t)$

### 연습 문제 3: 시스템 분석

인과적 LTI 시스템이 다음의 차분 방정식으로 기술됩니다:

$$y[n] = 0.5 y[n-1] + x[n]$$

1. $x[n] = \delta[n]$을 설정하고 반복하여 임펄스 응답 $h[n]$을 구하세요
2. 시스템이 BIBO 안정한가? 답을 증명하세요
3. 계단 응답을 계산하세요
4. 주파수 응답 $H(e^{j\omega})$를 구하세요
5. Python을 사용하여 크기 응답과 위상 응답을 그래프로 그리세요

### 연습 문제 4: 직렬 연결 vs. 병렬 연결

두 LTI 시스템의 임펄스 응답이 다음과 같습니다:

$$h_1[n] = (0.7)^n u[n], \qquad h_2[n] = (0.5)^n u[n]$$

1. 직렬 연결 $h_1 * h_2$의 임펄스 응답을 구하세요
2. 병렬 연결 $h_1 + h_2$의 임펄스 응답을 구하세요
3. 직렬 연결 순서가 중요하지 않음을 수치적으로 검증하세요: $h_1 * h_2 = h_2 * h_1$
4. 입력 $x[n] = \delta[n] - 0.3\delta[n-1]$에 대해, 두 구성 모두에서 출력을 계산하고 그래프로 그리세요

### 연습 문제 5: 이동 평균 필터 분석

$M$점 이동 평균 필터 $h[n] = \frac{1}{M}$ ($n = 0, 1, \ldots, M-1$)에 대해:

1. 닫힌 형태로 주파수 응답 $H(e^{j\omega})$를 유도하세요
2. 영점(null)이 발생하는 주파수는 어디인가? ($H$의 영점)
3. $M$의 함수로 3-dB 대역폭은 얼마인가?
4. $M = 3, 7, 15, 31$의 크기 응답을 같은 그래프에 그리세요
5. $f_s = 1000$ Hz에서 50 Hz와 400 Hz 성분을 포함하는 신호에 각 필터를 적용하세요. 어느 $M$이 두 성분을 가장 잘 분리하는가?

### 연습 문제 6: 실제 합성곱

1. $f_s = 8000$ Hz에서 100 Hz부터 2000 Hz까지 스윕하는 1초 처프(chirp) 신호를 생성하세요
2. 50ms, 120ms, 200ms에서 감소하는 크기를 가진 세 개의 반사가 있는 에코 시스템을 만드세요
3. 처프를 에코 시스템과 합성곱하세요
4. 직접 합성곱 시간과 FFT 합성곱 시간을 비교하세요
5. 입력과 출력의 스펙트로그램(spectrogram)을 나란히 그리세요

### 연습 문제 7: 시스템 식별

미지의 LTI 시스템(블랙박스)에 접근할 수 있습니다. 임의의 입력을 인가하고 출력을 관찰할 수 있습니다.

1. 단위 임펄스를 인가하여 $h[n]$을 직접 측정하세요
2. 단위 계단을 인가하여 $s[n]$을 측정한 후, $h[n] = s[n] - s[n-1]$로 $h[n]$을 복원하세요
3. 길이 10000의 백색 잡음을 인가하고, 교차 상관(cross-correlation)을 사용하여 $h[n]$을 추정하세요
4. 세 방법을 비교하세요. 어느 방법이 가장 적은 신호 에너지로 가장 정확한 결과를 제공하는가?

다음 블랙박스 시스템을 사용하여 구현하세요 (테스트 시 구현 내용을 보지 마세요):

```python
def black_box_system(x):
    """Unknown LTI system — treat as a black box."""
    h = np.array([0.2, 0.5, 1.0, 0.5, 0.2, -0.1, -0.05])
    return np.convolve(x, h, mode='full')[:len(x)]
```

### 연습 문제 8: 역합성곱

$y[n] = x[n] * h[n]$에서 $h[n]$이 알려져 있고 $y[n]$이 측정된 경우, $x[n]$을 복원하는 것을 **역합성곱(deconvolution)**이라 합니다.

1. 100개 샘플 중 5개의 비영 값을 가진 희소 신호 $x[n]$을 생성하세요
2. $h[n] = [1, 0.5, 0.25]$와 합성곱하여 $y[n]$을 얻으세요
3. 소량의 잡음 추가: $y_{\text{noisy}}[n] = y[n] + 0.01 \cdot w[n]$ ($w$는 백색 잡음)
4. FFT를 사용하여 역합성곱을 시도하세요: 주파수 영역에서 $X = Y/H$
5. 직접 역합성곱이 문제가 되는 이유를 설명하세요 (힌트: $H \approx 0$인 주파수 고려)
6. 위너 역합성곱(Wiener deconvolution)을 구현하세요: $\hat{X} = \frac{H^* Y}{|H|^2 + \lambda}$, 그리고 개선 효과를 보이세요

---

## 14. 참고문헌

1. Oppenheim, A. V. & Willsky, A. S. *Signals and Systems* (2판), Ch. 2-3. Prentice Hall, 1997.
2. Oppenheim, A. V. & Schafer, R. W. *Discrete-Time Signal Processing* (3판), Ch. 2. Pearson, 2010.
3. Haykin, S. & Van Veen, B. *Signals and Systems* (2판), Ch. 2-4. Wiley, 2003.
4. Smith, S. W. *The Scientist and Engineer's Guide to Digital Signal Processing*, Ch. 6-7. California Technical Publishing, 1997. (무료 온라인: dspguide.com)

---

[이전: 01. 신호와 시스템](./01_Signals_and_Systems.md) | [다음: 03. 푸리에 급수와 응용](./03_Fourier_Series_and_Applications.md) | [개요](./00_Overview.md)
