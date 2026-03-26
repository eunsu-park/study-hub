# 08. 푸리에 변환 (Fourier Transforms)

## 학습 목표

- **푸리에 급수에서 푸리에 변환으로**의 전환 과정을 이해하고, 비주기 함수에 대한 스펙트럼 표현을 유도할 수 있다
- **푸리에 변환의 주요 성질** (선형성, 이동, 스케일링, 미분)을 증명하고 활용할 수 있다
- **가우시안, 직사각형 함수, 디랙 델타 함수** 등 핵심 변환 쌍을 계산하고 물리적으로 해석할 수 있다
- **컨볼루션 정리**를 이해하고, 신호 필터링 등 실제 응용에 적용할 수 있다
- **이산 푸리에 변환(DFT)과 FFT** 알고리즘의 원리를 이해하고, 나이퀴스트 정리의 의미를 설명할 수 있다
- **불확정성 원리, 프라운호퍼 회절, 스펙트럼 분석** 등 물리학 응용에 푸리에 변환을 적용할 수 있다

---

## 1. 푸리에 급수에서 푸리에 변환으로

### 1.1 주기 → 비주기 함수로의 일반화

주기 $T$를 가지는 함수 $f(x)$의 푸리에 급수는 다음과 같이 쓸 수 있습니다:

$$f(x) = \sum_{n=-\infty}^{\infty} c_n e^{i \cdot 2\pi n x / T}$$

여기서 복소 푸리에 계수 $c_n$은:

$$c_n = \frac{1}{T} \int_{-T/2}^{T/2} f(x) e^{-i \cdot 2\pi n x / T} \, dx$$

기본 주파수(fundamental frequency)를 $\omega_0 = 2\pi/T$, 주파수 간격을 $\Delta\omega = \omega_0 = 2\pi/T$로 정의하면:

$$f(x) = \sum_{n=-\infty}^{\infty} c_n e^{i n \omega_0 x}$$

$$c_n = \frac{1}{T} \int_{-T/2}^{T/2} f(x) e^{-i n \omega_0 x} \, dx$$

**핵심 아이디어**: 주기 $T$를 무한대로 보내면 ($T \to \infty$), 이산 주파수 $n\omega_0$가 연속 변수 $\omega$가 되고, 합(sum)이 적분(integral)으로 바뀝니다.

$$T \to \infty \quad \Rightarrow \quad \Delta\omega \to 0$$
$$n \omega_0 \to \omega \quad \text{(연속 변수)}$$
$$\sum \to \int$$

이 극한 과정에서 $c_n T$를 새로운 함수 $F(\omega)$로 정의하면:

$$F(\omega) = \lim_{T \to \infty} c_n T = \int_{-\infty}^{\infty} f(x) e^{-i\omega x} \, dx$$

이것이 바로 **푸리에 변환**(Fourier transform)입니다.

### 1.2 연속 푸리에 변환 정의

**푸리에 변환 (Fourier Transform)**:

$$F(\omega) = \int_{-\infty}^{\infty} f(x) e^{-i\omega x} \, dx$$

**역 푸리에 변환 (Inverse Fourier Transform)**:

$$f(x) = \frac{1}{2\pi} \int_{-\infty}^{\infty} F(\omega) e^{i\omega x} \, d\omega$$

> **관례에 대하여**: 교재와 분야에 따라 $2\pi$ 인자의 배치가 다릅니다. Boas 교재는 위 관례를 따르며, 물리학에서는 대칭적 관례 (양쪽에 $1/\sqrt{2\pi}$)를 쓰기도 합니다. 어느 관례를 쓰든 변환-역변환 쌍이 일관되기만 하면 됩니다.

**존재 조건**: $f(x)$가 절대적분 가능(absolutely integrable)하면 푸리에 변환이 존재합니다:

$$\int_{-\infty}^{\infty} |f(x)| \, dx < \infty$$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft as sp_fft

# 예제: 주기 T를 증가시키면서 이산 스펙트럼 -> 연속 스펙트럼으로의 전환 관찰

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
periods = [2, 5, 20]

for idx, T in enumerate(periods):
    # 유한 구간에서의 가우시안 함수
    x = np.linspace(-T/2, T/2, 1000)
    sigma = 0.5
    f_x = np.exp(-x**2 / (2 * sigma**2))

    # 시간 영역
    axes[0, idx].plot(x, f_x, 'b-', linewidth=1.5)
    axes[0, idx].set_title(f'f(x), T = {T}')
    axes[0, idx].set_xlabel('x')
    axes[0, idx].set_xlim(-5, 5)
    axes[0, idx].grid(True, alpha=0.3)

    # 푸리에 계수 (이산 스펙트럼)
    n_max = 30
    n_vals = np.arange(-n_max, n_max + 1)
    omega_0 = 2 * np.pi / T
    c_n = []
    for n in n_vals:
        integrand = f_x * np.exp(-1j * n * omega_0 * x)
        cn = np.trapz(integrand, x) / T
        c_n.append(np.abs(cn))

    omega_vals = n_vals * omega_0
    axes[1, idx].stem(omega_vals, c_n, linefmt='b-', markerfmt='bo', basefmt='k-')
    axes[1, idx].set_title(f'|c_n|, T = {T}')
    axes[1, idx].set_xlabel('omega')
    axes[1, idx].set_xlim(-15, 15)
    axes[1, idx].grid(True, alpha=0.3)

plt.suptitle('주기 T 증가 → 이산 스펙트럼이 연속 스펙트럼에 접근', fontsize=14)
plt.tight_layout()
plt.savefig('fourier_discrete_to_continuous.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 2. 푸리에 변환의 성질

### 2.1 선형성과 대칭성

**선형성 (Linearity)**: 푸리에 변환은 선형 연산자(linear operator)입니다:

$$\mathcal{F}[a f(x) + b g(x)] = a F(\omega) + b G(\omega)$$

여기서 $a, b$는 상수이고, $F(\omega) = \mathcal{F}[f]$, $G(\omega) = \mathcal{F}[g]$입니다.

**대칭성 (Symmetry / Duality)**:

$f(x)$의 푸리에 변환이 $F(\omega)$이면:

$$\mathcal{F}[F(x)] = 2\pi f(-\omega)$$

즉, 시간 영역과 주파수 영역의 역할을 바꾸면, 원래 함수를 (반전시켜) 되돌려 받습니다.

**실함수의 대칭성**: $f(x)$가 실수 함수이면:

$$F(-\omega) = F^*(\omega) \quad \text{(켤레 대칭, Hermitian symmetry)}$$

$$|F(-\omega)| = |F(\omega)| \quad \text{(크기 스펙트럼은 짝함수)}$$

### 2.2 시간 이동과 주파수 이동

**시간 이동 (Time Shift)**: 함수를 $x_0$만큼 이동하면, 주파수 영역에서는 위상(phase)만 변합니다:

$$\mathcal{F}[f(x - x_0)] = e^{-i\omega x_0} F(\omega)$$

크기 스펙트럼 $|F(\omega)|$는 변하지 않고, 위상만 $\omega x_0$만큼 바뀝니다.

**주파수 이동 (Frequency Shift / Modulation)**: 주파수 영역에서 $\omega_0$만큼 이동:

$$\mathcal{F}[f(x) e^{i\omega_0 x}] = F(\omega - \omega_0)$$

이것은 통신에서의 **변조(modulation)** 원리입니다. 반송파 주파수 $\omega_0$를 곱하면, 스펙트럼이 $\omega_0$만큼 이동합니다.

### 2.3 스케일링 정리

함수의 시간 축을 $a$배 압축/확장하면:

$$\mathcal{F}[f(ax)] = \frac{1}{|a|} F\left(\frac{\omega}{a}\right)$$

**물리적 의미**: 시간 영역에서 신호가 좁아지면($a > 1$), 주파수 영역에서는 넓어집니다. 이것이 **불확정성 원리**의 수학적 근간입니다.

```python
import numpy as np
import matplotlib.pyplot as plt

# 스케일링 정리 시각화: 좁은 펄스 ↔ 넓은 스펙트럼

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
a_values = [0.5, 1.0, 2.0]

x = np.linspace(-5, 5, 1000)
dx = x[1] - x[0]
omega = np.linspace(-10, 10, 1000)

for idx, a in enumerate(a_values):
    # 가우시안: f(ax) = exp(-(ax)^2 / 2)
    f = np.exp(-(a * x)**2 / 2)

    # 해석적 푸리에 변환: (1/|a|) * sqrt(2*pi) * exp(-omega^2 / (2*a^2))
    F_analytical = (1 / abs(a)) * np.sqrt(2 * np.pi) * np.exp(-omega**2 / (2 * a**2))

    axes[0, idx].plot(x, f, 'b-', linewidth=2)
    axes[0, idx].set_title(f'f({a}x) = exp(-({a}x)²/2)')
    axes[0, idx].set_xlabel('x')
    axes[0, idx].set_ylim(-0.1, 1.5)
    axes[0, idx].grid(True, alpha=0.3)

    axes[1, idx].plot(omega, F_analytical, 'r-', linewidth=2)
    axes[1, idx].set_title(f'F(omega), a = {a}')
    axes[1, idx].set_xlabel('omega')
    axes[1, idx].set_ylim(-0.1, 6)
    axes[1, idx].grid(True, alpha=0.3)

plt.suptitle('스케일링 정리: 좁은 펄스 ↔ 넓은 스펙트럼', fontsize=14)
plt.tight_layout()
plt.savefig('scaling_theorem.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 2.4 미분과 적분의 변환

**미분 정리**: 시간 영역의 미분은 주파수 영역에서 $i\omega$를 곱하는 것에 대응합니다:

$$\mathcal{F}[f'(x)] = i\omega F(\omega)$$

$$\mathcal{F}[f^{(n)}(x)] = (i\omega)^n F(\omega)$$

이 성질 덕분에 미분방정식을 대수방정식으로 바꿀 수 있습니다.

**예제**: 열 전도 방정식 (heat equation)

$$\frac{\partial u}{\partial t} = k \frac{\partial^2 u}{\partial x^2}$$

양변에 $x$에 대한 푸리에 변환을 적용하면:

$$\frac{dU}{dt} = k (i\omega)^2 U = -k\omega^2 U$$

이것은 $\omega$를 매개변수로 가지는 일반 미분방정식(ODE)이며, 해는:

$$U(\omega, t) = U(\omega, 0) e^{-k\omega^2 t}$$

**주파수 영역 미분**: 주파수 영역에서의 미분은 시간 영역에서 $-ix$를 곱하는 것에 대응합니다:

$$\mathcal{F}[-i x f(x)] = F'(\omega)$$

$$\mathcal{F}[(-ix)^n f(x)] = F^{(n)}(\omega)$$

---

## 3. 중요한 변환 쌍

### 3.1 가우시안 함수

가우시안 함수의 푸리에 변환은 다시 가우시안입니다. 이 성질이 가우시안을 특별하게 만듭니다:

$$f(x) = e^{-ax^2} \quad (a > 0)$$

$$F(\omega) = \sqrt{\frac{\pi}{a}} e^{-\omega^2 / (4a)}$$

**증명 스케치**: 적분 안에서 지수를 완전제곱식(completing the square)으로 만들면:

$$F(\omega) = \int_{-\infty}^{\infty} e^{-ax^2} e^{-i\omega x} \, dx$$

$$= \int_{-\infty}^{\infty} e^{-a(x + i\omega/(2a))^2 - \omega^2/(4a)} \, dx$$

$$= e^{-\omega^2/(4a)} \sqrt{\frac{\pi}{a}}$$

마지막 단계에서 가우스 적분 $\int e^{-au^2} \, du = \sqrt{\pi/a}$를 사용했습니다.

> **핵심**: 가우시안은 푸리에 변환의 **고유함수(eigenfunction)** 입니다 — 변환해도 형태가 보존됩니다. 시간 영역에서의 폭 $\sigma_x = 1/\sqrt{2a}$와 주파수 영역에서의 폭 $\sigma_\omega = \sqrt{2a}$의 곱은 항상 일정합니다: $\sigma_x \sigma_\omega = 1$.

### 3.2 직사각형 함수 (rect)와 sinc 함수

**직사각형 함수**:

$$\text{rect}(x/a) = \begin{cases} 1, & |x| < a/2 \\ 0, & |x| > a/2 \end{cases}$$

**푸리에 변환**:

$$\mathcal{F}[\text{rect}(x/a)] = a \cdot \text{sinc}(\omega a / (2\pi)) = a \cdot \frac{\sin(\omega a/2)}{\omega a/2}$$

여기서 sinc 함수는 $\text{sinc}(u) = \sin(\pi u) / (\pi u)$입니다.

```python
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
widths = [1.0, 2.0, 4.0]

for idx, a in enumerate(widths):
    # 직사각형 함수
    x = np.linspace(-5, 5, 1000)
    rect = np.where(np.abs(x) < a / 2, 1.0, 0.0)

    # 해석적 푸리에 변환: a * sin(omega*a/2) / (omega*a/2)
    omega = np.linspace(-20, 20, 1000)
    # omega = 0일 때 특이점 처리
    with np.errstate(divide='ignore', invalid='ignore'):
        F_omega = a * np.sinc(omega * a / (2 * np.pi))

    axes[0, idx].plot(x, rect, 'b-', linewidth=2)
    axes[0, idx].set_title(f'rect(x/{a}), 폭 = {a}')
    axes[0, idx].set_xlabel('x')
    axes[0, idx].set_ylim(-0.2, 1.5)
    axes[0, idx].grid(True, alpha=0.3)

    axes[1, idx].plot(omega, F_omega, 'r-', linewidth=1.5)
    axes[1, idx].set_title(f'F(omega), a = {a}')
    axes[1, idx].set_xlabel('omega')
    axes[1, idx].grid(True, alpha=0.3)

plt.suptitle('직사각형 함수 ↔ sinc 함수: 폭이 넓을수록 스펙트럼이 좁음', fontsize=14)
plt.tight_layout()
plt.savefig('rect_sinc_pair.png', dpi=150, bbox_inches='tight')
plt.show()
```

**물리적 의미**: 좁은 슬릿(rect 폭이 작음)은 넓은 회절 패턴(sinc가 넓어짐)을 만들고, 넓은 슬릿은 좁은 회절 패턴을 만듭니다. 이것이 프라운호퍼 회절의 핵심입니다.

### 3.3 디랙 델타 함수

**디랙 델타 함수 (Dirac delta function)** $\delta(x)$는 다음을 만족하는 분포(distribution)입니다:

$$\int_{-\infty}^{\infty} \delta(x) g(x) \, dx = g(0)$$

$$\delta(x) = 0 \quad (x \neq 0)$$

$$\int_{-\infty}^{\infty} \delta(x) \, dx = 1$$

**디랙 델타의 푸리에 변환**:

$$\mathcal{F}[\delta(x)] = \int_{-\infty}^{\infty} \delta(x) e^{-i\omega x} \, dx = 1$$

즉, 디랙 델타의 스펙트럼은 모든 주파수에 균일하게 퍼져 있습니다 — **백색 스펙트럼(white spectrum)**.

**역으로**: 상수 함수 1의 푸리에 변환은 $2\pi\delta(\omega)$입니다:

$$\mathcal{F}[1] = 2\pi \delta(\omega)$$

> **물리적 의미**: 시간 영역에서 무한히 짧은 충격(impulse)은 모든 주파수 성분을 동일하게 포함합니다. 반대로, 영원히 지속되는 순수한 톤(상수)은 단 하나의 주파수만 가집니다.

**이동된 델타 함수**:

$$\mathcal{F}[\delta(x - x_0)] = e^{-i\omega x_0}$$

$$\mathcal{F}[e^{i\omega_0 x}] = 2\pi \delta(\omega - \omega_0)$$

### 3.4 주요 변환 쌍 표

| $f(x)$ | $F(\omega)$ | 비고 |
|------|----------|------|
| $e^{-a|x|}$ | $\frac{2a}{a^2 + \omega^2}$ | 로렌츠 함수 (Lorentzian) |
| $e^{-ax^2}$ | $\sqrt{\pi/a} \cdot e^{-\omega^2/(4a)}$ | 가우시안 (Gaussian) |
| $\text{rect}(x/a)$ | $a \cdot \text{sinc}(\omega a/(2\pi))$ | 직사각형 ↔ sinc |
| $\delta(x)$ | $1$ | 임펄스 ↔ 백색 |
| $1$ | $2\pi\delta(\omega)$ | 상수 ↔ DC |
| $e^{i\omega_0 x}$ | $2\pi\delta(\omega - \omega_0)$ | 복소 지수 |
| $\cos(\omega_0 x)$ | $\pi[\delta(\omega-\omega_0) + \delta(\omega+\omega_0)]$ | 코사인 |
| $x^n e^{-a|x|}$ | $\frac{d^n}{d\omega^n}\left[\frac{2a}{a^2+\omega^2}\right]$ 변형 | 감쇠 다항식 |
| $\frac{1}{x^2 + a^2}$ | $\frac{\pi}{a} e^{-a|\omega|}$ | 로렌츠 역변환 |
| $e^{-ax} u(x)$ | $\frac{1}{a + i\omega}$ | 단측 지수감쇠, $u(x)$=계단함수 |

---

## 4. 컨볼루션 정리

### 4.1 컨볼루션 (Convolution)의 정의

두 함수 $f(x)$와 $g(x)$의 **컨볼루션(convolution)** 은 다음과 같이 정의됩니다:

$$(f * g)(x) = \int_{-\infty}^{\infty} f(t) g(x - t) \, dt$$

기하학적으로, $g$를 반전시킨 뒤 $f$ 위로 미끄러뜨리면서 겹치는 면적을 계산하는 연산입니다.

**컨볼루션의 교환법칙**: $f * g = g * f$

### 4.2 시간 영역 컨볼루션 ↔ 주파수 영역 곱

**컨볼루션 정리 (Convolution Theorem)**:

$$\mathcal{F}[f * g] = F(\omega) G(\omega)$$

즉, **시간 영역에서의 컨볼루션은 주파수 영역에서의 곱셈**에 대응합니다.

**역도 성립합니다**:

$$\mathcal{F}[f \cdot g] = \frac{1}{2\pi} F(\omega) * G(\omega) \quad \text{(곱의 변환)}$$

시간 영역에서의 곱셈은 주파수 영역에서의 컨볼루션에 대응합니다.

**증명**:

$$\mathcal{F}[f * g] = \int_{-\infty}^{\infty} \left[\int_{-\infty}^{\infty} f(t) g(x-t) \, dt\right] e^{-i\omega x} \, dx$$

적분 순서를 교환하고 $u = x - t$로 치환하면:

$$= \int_{-\infty}^{\infty} f(t) e^{-i\omega t} \, dt \cdot \int_{-\infty}^{\infty} g(u) e^{-i\omega u} \, du = F(\omega) G(\omega)$$

### 4.3 응용: 필터링

컨볼루션 정리의 가장 중요한 응용은 **신호 필터링(signal filtering)** 입니다.

입력 신호 $x(t)$를 필터 $h(t)$에 통과시키면:

$$y(t) = (x * h)(t) \quad \text{(시간 영역: 컨볼루션)}$$

$$Y(\omega) = X(\omega) H(\omega) \quad \text{(주파수 영역: 곱셈)}$$

$H(\omega)$를 **전달 함수(transfer function)** 또는 **주파수 응답(frequency response)** 이라 합니다.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

# 컨볼루션 정리 시각화: 신호 필터링

np.random.seed(42)
N = 1024
dt = 0.01
t = np.arange(N) * dt

# 원본 신호: 깨끗한 사인파 + 잡음
freq_signal = 5.0  # 5 Hz 신호
signal_clean = np.sin(2 * np.pi * freq_signal * t)
noise = 0.5 * np.random.randn(N)
signal_noisy = signal_clean + noise

# 가우시안 저역통과 필터 (Low-pass filter)
sigma_filter = 0.05
t_filter = np.arange(-50, 51) * dt
h = np.exp(-t_filter**2 / (2 * sigma_filter**2))
h /= h.sum()  # 정규화

# 시간 영역 컨볼루션
signal_filtered = fftconvolve(signal_noisy, h, mode='same')

# 주파수 영역 확인
freqs = np.fft.fftfreq(N, dt)
X_noisy = np.fft.fft(signal_noisy)
H_freq = np.fft.fft(h, N)  # 필터를 같은 길이로 FFT
Y_freq = X_noisy * H_freq

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# 시간 영역
axes[0, 0].plot(t, signal_noisy, 'gray', alpha=0.7, label='잡음 신호')
axes[0, 0].plot(t, signal_clean, 'b-', linewidth=2, label='원본 신호')
axes[0, 0].set_title('입력 신호 (시간 영역)')
axes[0, 0].set_xlabel('t (s)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[1, 0].plot(t_filter, h, 'g-', linewidth=2)
axes[1, 0].set_title('가우시안 필터 h(t)')
axes[1, 0].set_xlabel('t (s)')
axes[1, 0].grid(True, alpha=0.3)

axes[2, 0].plot(t, signal_filtered, 'r-', linewidth=2, label='필터링 결과')
axes[2, 0].plot(t, signal_clean, 'b--', linewidth=1, label='원본 신호')
axes[2, 0].set_title('출력 신호 y(t) = (x * h)(t)')
axes[2, 0].set_xlabel('t (s)')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

# 주파수 영역
mask = freqs >= 0
axes[0, 1].plot(freqs[mask], np.abs(X_noisy[mask]) / N, 'gray', alpha=0.7)
axes[0, 1].set_title('|X(omega)| 입력 스펙트럼')
axes[0, 1].set_xlabel('주파수 (Hz)')
axes[0, 1].set_xlim(0, 50)
axes[0, 1].grid(True, alpha=0.3)

axes[1, 1].plot(freqs[mask], np.abs(H_freq[mask]), 'g-', linewidth=2)
axes[1, 1].set_title('|H(omega)| 필터 주파수 응답')
axes[1, 1].set_xlabel('주파수 (Hz)')
axes[1, 1].set_xlim(0, 50)
axes[1, 1].grid(True, alpha=0.3)

axes[2, 1].plot(freqs[mask], np.abs(Y_freq[mask]) / N, 'r-', linewidth=2)
axes[2, 1].set_title('|Y(omega)| = |X(omega)| * |H(omega)|')
axes[2, 1].set_xlabel('주파수 (Hz)')
axes[2, 1].set_xlim(0, 50)
axes[2, 1].grid(True, alpha=0.3)

plt.suptitle('컨볼루션 정리: 시간 영역 컨볼루션 = 주파수 영역 곱셈', fontsize=14)
plt.tight_layout()
plt.savefig('convolution_theorem.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 5. 이산 푸리에 변환 (DFT)과 FFT

### 5.1 연속 → 이산으로

실제 측정 데이터는 연속적이지 않고, 일정 간격 $\Delta t$로 **샘플링(sampling)** 됩니다:

$$x_n = f(n \Delta t), \quad n = 0, 1, 2, \ldots, N-1$$

총 $N$개의 샘플을 가지며, 전체 관측 시간은 $T = N \Delta t$입니다.

연속 푸리에 변환의 적분을 이산 합으로 근사하면 DFT를 얻습니다.

### 5.2 DFT의 정의

**이산 푸리에 변환 (Discrete Fourier Transform, DFT)**:

$$X_k = \sum_{n=0}^{N-1} x_n e^{-i \cdot 2\pi k n / N}, \quad k = 0, 1, \ldots, N-1$$

**역 이산 푸리에 변환 (Inverse DFT)**:

$$x_n = \frac{1}{N} \sum_{k=0}^{N-1} X_k e^{i \cdot 2\pi k n / N}, \quad n = 0, 1, \ldots, N-1$$

**주파수 해상도**: $k$번째 주파수 성분에 대응하는 물리적 주파수는:

$$f_k = \frac{k}{N \Delta t} = \frac{k}{T}$$

주파수 해상도 $\Delta f = 1/T$는 관측 시간에 반비례합니다.

### 5.3 FFT 알고리즘 개요

DFT를 직접 계산하면 $O(N^2)$의 연산이 필요합니다. **고속 푸리에 변환 (Fast Fourier Transform, FFT)** 은 이를 $O(N \log N)$으로 줄입니다.

**Cooley-Tukey 알고리즘** (1965)의 핵심 아이디어:

1. $N$개의 점에 대한 DFT를 $N/2$개의 짝수 인덱스(even)와 $N/2$개의 홀수 인덱스(odd) DFT로 분할:

$$X_k = \sum_{\text{even } n} x_n W^{kn} + \sum_{\text{odd } n} x_n W^{kn} = E_k + W^k O_k$$

여기서 $W = e^{-i \cdot 2\pi / N}$ (회전 인자, twiddle factor)

2. 이 과정을 재귀적으로 반복하면 $\log_2(N)$ 단계만에 완료됩니다.

```python
import numpy as np
import time
import matplotlib.pyplot as plt

# DFT vs FFT 계산 시간 비교

def dft_naive(x):
    """순진한 DFT 구현: O(N^2)"""
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

# 다양한 크기에서 시간 측정
sizes = [64, 128, 256, 512, 1024]
times_dft = []
times_fft = []

for N in sizes:
    x = np.random.randn(N)

    # DFT (순진한 구현)
    t_start = time.perf_counter()
    X_dft = dft_naive(x)
    t_dft = time.perf_counter() - t_start
    times_dft.append(t_dft)

    # FFT (NumPy)
    t_start = time.perf_counter()
    X_fft = np.fft.fft(x)
    t_fft = time.perf_counter() - t_start
    times_fft.append(t_fft)

    # 결과 검증
    error = np.max(np.abs(X_dft - X_fft))
    print(f"N={N:5d}: DFT={t_dft:.4f}s, FFT={t_fft:.6f}s, "
          f"speedup={t_dft/t_fft:.0f}x, max_error={error:.2e}")

plt.figure(figsize=(10, 6))
plt.loglog(sizes, times_dft, 'ro-', label='DFT O(N²)', markersize=8)
plt.loglog(sizes, times_fft, 'bs-', label='FFT O(N log N)', markersize=8)
plt.xlabel('N (데이터 크기)')
plt.ylabel('계산 시간 (초)')
plt.title('DFT vs FFT 계산 시간 비교')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('dft_vs_fft_timing.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 5.4 나이퀴스트 정리와 앨리어싱

**나이퀴스트-섀넌 샘플링 정리 (Nyquist-Shannon Sampling Theorem)**:

> 신호에 포함된 최대 주파수가 $f_{\max}$일 때, 원래 신호를 완벽하게 복원하려면 샘플링 주파수 $f_s$는 최소 $2 f_{\max}$ 이상이어야 합니다.

$$f_s \geq 2 f_{\max} \quad \text{(나이퀴스트 조건)}$$

$$f_{\text{Nyquist}} = \frac{f_s}{2} \quad \text{(나이퀴스트 주파수: 표현 가능한 최대 주파수)}$$

**앨리어싱 (Aliasing)**: 나이퀴스트 조건을 만족하지 못하면, 고주파 성분이 저주파로 잘못 접혀(fold) 나타납니다.

```python
import numpy as np
import matplotlib.pyplot as plt

# 앨리어싱 시각화

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 원본 연속 신호: 10 Hz 사인파
f_signal = 10.0  # Hz
t_continuous = np.linspace(0, 1, 10000)
signal_continuous = np.sin(2 * np.pi * f_signal * t_continuous)

# 케이스 1: 충분한 샘플링 (f_s = 50 Hz > 2 * 10 Hz)
f_s_good = 50.0
t_good = np.arange(0, 1, 1/f_s_good)
signal_good = np.sin(2 * np.pi * f_signal * t_good)

# 케이스 2: 부족한 샘플링 (f_s = 12 Hz < 2 * 10 Hz)
f_s_bad = 12.0
t_bad = np.arange(0, 1, 1/f_s_bad)
signal_bad = np.sin(2 * np.pi * f_signal * t_bad)

# 앨리어싱된 주파수: |f_signal - f_s_bad| = |10 - 12| = 2 Hz
f_alias = abs(f_signal - f_s_bad)
signal_alias = np.sin(2 * np.pi * f_alias * t_continuous)

# 시간 영역: 충분한 샘플링
axes[0, 0].plot(t_continuous, signal_continuous, 'b-', alpha=0.3, label=f'원본 {f_signal} Hz')
axes[0, 0].stem(t_good, signal_good, linefmt='b-', markerfmt='bo', basefmt='k-')
axes[0, 0].set_title(f'충분한 샘플링: f_s = {f_s_good} Hz')
axes[0, 0].set_xlabel('t (s)')
axes[0, 0].set_xlim(0, 0.5)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 시간 영역: 부족한 샘플링
axes[0, 1].plot(t_continuous, signal_continuous, 'b-', alpha=0.3, label=f'원본 {f_signal} Hz')
axes[0, 1].plot(t_continuous, signal_alias, 'r--', alpha=0.5, label=f'앨리어스 {f_alias} Hz')
axes[0, 1].stem(t_bad, signal_bad, linefmt='r-', markerfmt='ro', basefmt='k-')
axes[0, 1].set_title(f'부족한 샘플링: f_s = {f_s_bad} Hz (앨리어싱 발생!)')
axes[0, 1].set_xlabel('t (s)')
axes[0, 1].set_xlim(0, 0.5)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 주파수 영역: 충분한 샘플링
N_good = len(signal_good)
freqs_good = np.fft.fftfreq(N_good, 1/f_s_good)
X_good = np.fft.fft(signal_good) / N_good
mask_good = freqs_good >= 0
axes[1, 0].stem(freqs_good[mask_good], 2 * np.abs(X_good[mask_good]),
                linefmt='b-', markerfmt='bo', basefmt='k-')
axes[1, 0].axvline(x=f_s_good/2, color='g', linestyle='--', label=f'f_Nyquist = {f_s_good/2} Hz')
axes[1, 0].set_title('스펙트럼 (충분한 샘플링)')
axes[1, 0].set_xlabel('주파수 (Hz)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 주파수 영역: 부족한 샘플링
N_bad = len(signal_bad)
freqs_bad = np.fft.fftfreq(N_bad, 1/f_s_bad)
X_bad = np.fft.fft(signal_bad) / N_bad
mask_bad = freqs_bad >= 0
axes[1, 1].stem(freqs_bad[mask_bad], 2 * np.abs(X_bad[mask_bad]),
                linefmt='r-', markerfmt='ro', basefmt='k-')
axes[1, 1].axvline(x=f_s_bad/2, color='g', linestyle='--', label=f'f_Nyquist = {f_s_bad/2} Hz')
axes[1, 1].set_title('스펙트럼 (부족한 샘플링 → 앨리어싱)')
axes[1, 1].set_xlabel('주파수 (Hz)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('나이퀴스트 정리와 앨리어싱', fontsize=14)
plt.tight_layout()
plt.savefig('nyquist_aliasing.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 6. 물리학 응용

### 6.1 불확정성 원리 (시간-주파수 관계)

**수학적 불확정성 원리**: 임의의 함수 $f(x)$와 그 푸리에 변환 $F(\omega)$에 대해:

$$\Delta x \cdot \Delta\omega \geq \frac{1}{2}$$

여기서 $\Delta x$, $\Delta\omega$는 각각 시간/주파수 영역에서의 표준편차(standard deviation)로 정의됩니다.

**양자역학과의 연결**: 드브로이 관계 $p = \hbar k$ ($k = 2\pi/\lambda$)에 의해, 위치-운동량 불확정성 원리가 됩니다:

$$\Delta x \cdot \Delta p \geq \frac{\hbar}{2}$$

이것은 양자역학의 기본 원리이지만, 그 수학적 본질은 **푸리에 변환의 성질**입니다.

```python
import numpy as np
import matplotlib.pyplot as plt

# 불확정성 원리 시각화: 가우시안 파동 패킷

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
sigmas = [0.5, 1.0, 2.0]  # 위치 공간에서의 폭

x = np.linspace(-10, 10, 2000)
k = np.linspace(-10, 10, 2000)

for idx, sigma_x in enumerate(sigmas):
    # 위치 공간 파동함수 (가우시안 파동 패킷)
    k0 = 3.0  # 중심 파수
    psi_x = (1 / (2 * np.pi * sigma_x**2)**0.25) * \
            np.exp(-x**2 / (4 * sigma_x**2)) * np.exp(1j * k0 * x)

    # 운동량 공간 파동함수 (푸리에 변환)
    sigma_k = 1 / (2 * sigma_x)  # 운동량 공간의 폭
    psi_k = (2 * sigma_x**2 / np.pi)**0.25 * \
            np.exp(-(k - k0)**2 * sigma_x**2)

    # 불확정성 곱
    product = sigma_x * sigma_k

    # 위치 공간
    axes[0, idx].plot(x, np.abs(psi_x)**2, 'b-', linewidth=2, label=f'|psi(x)|^2')
    axes[0, idx].fill_between(x, np.abs(psi_x)**2, alpha=0.2, color='blue')
    axes[0, idx].axvspan(-sigma_x, sigma_x, alpha=0.1, color='red', label=f'sigma_x = {sigma_x}')
    axes[0, idx].set_title(f'위치 공간: sigma_x = {sigma_x}')
    axes[0, idx].set_xlabel('x')
    axes[0, idx].legend(fontsize=9)
    axes[0, idx].grid(True, alpha=0.3)

    # 운동량 공간
    axes[1, idx].plot(k, np.abs(psi_k)**2, 'r-', linewidth=2, label=f'|psi(k)|^2')
    axes[1, idx].fill_between(k, np.abs(psi_k)**2, alpha=0.2, color='red')
    axes[1, idx].axvspan(k0 - sigma_k, k0 + sigma_k, alpha=0.1, color='blue',
                          label=f'sigma_k = {sigma_k:.2f}')
    axes[1, idx].set_title(f'운동량 공간: sigma_k = {sigma_k:.2f}\n'
                           f'sigma_x * sigma_k = {product:.2f} >= 0.5')
    axes[1, idx].set_xlabel('k')
    axes[1, idx].legend(fontsize=9)
    axes[1, idx].grid(True, alpha=0.3)

plt.suptitle('하이젠베르크 불확정성 원리: Delta_x * Delta_k >= 1/2', fontsize=14)
plt.tight_layout()
plt.savefig('uncertainty_principle.png', dpi=150, bbox_inches='tight')
plt.show()
```

> **가우시안 파동 패킷(Gaussian wave packet)** 은 불확정성 원리의 등호를 만족하는 유일한 함수입니다 (최소 불확정성 상태, minimum uncertainty state).

### 6.2 프라운호퍼 회절

단일 슬릿을 통과하는 빛의 **프라운호퍼 회절(Fraunhofer diffraction)** 패턴은 슬릿의 투과 함수(aperture function)의 **푸리에 변환**입니다:

$$U(\theta) \sim \mathcal{F}[A(x)]$$

여기서 $A(x)$는 슬릿의 형상(aperture function)이고, 관측 각도 $\theta$와 공간주파수의 관계는:

$$\omega = \frac{2\pi}{\lambda} \sin\theta$$

**단일 슬릿**: $A(x) = \text{rect}(x/a)$이므로:

$$U(\theta) \sim \text{sinc}\left(\frac{\pi a \sin\theta}{\lambda}\right)$$

$$I(\theta) \sim \text{sinc}^2\left(\frac{\pi a \sin\theta}{\lambda}\right)$$

```python
import numpy as np
import matplotlib.pyplot as plt

# 프라운호퍼 회절: 단일 슬릿과 이중 슬릿

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
wavelength = 500e-9  # 500 nm (초록색 빛)
k = 2 * np.pi / wavelength

theta = np.linspace(-0.05, 0.05, 2000)  # 관측 각도 (라디안)

# --- 단일 슬릿 ---
slit_widths = [10e-6, 20e-6, 50e-6]  # 10, 20, 50 마이크로미터

for idx, a in enumerate(slit_widths):
    beta = np.pi * a * np.sin(theta) / wavelength
    # sinc² 패턴 (beta=0에서 특이점 처리)
    with np.errstate(divide='ignore', invalid='ignore'):
        intensity = np.where(beta == 0, 1.0, (np.sin(beta) / beta)**2)

    axes[0, idx].plot(np.degrees(theta), intensity, 'b-', linewidth=1.5)
    axes[0, idx].set_title(f'단일 슬릿: a = {a*1e6:.0f} um')
    axes[0, idx].set_xlabel('theta (도)')
    axes[0, idx].set_ylabel('I / I_0')
    axes[0, idx].grid(True, alpha=0.3)

# --- 이중 슬릿 ---
a = 20e-6       # 슬릿 폭
d_values = [50e-6, 100e-6, 200e-6]  # 슬릿 간격

for idx, d in enumerate(d_values):
    beta = np.pi * a * np.sin(theta) / wavelength
    delta = np.pi * d * np.sin(theta) / wavelength

    with np.errstate(divide='ignore', invalid='ignore'):
        envelope = np.where(beta == 0, 1.0, (np.sin(beta) / beta)**2)
    interference = np.cos(delta)**2
    intensity = envelope * interference

    axes[1, idx].plot(np.degrees(theta), intensity, 'r-', linewidth=1, label='총 패턴')
    axes[1, idx].plot(np.degrees(theta), envelope, 'b--', linewidth=1, alpha=0.5, label='회절 엔벨로프')
    axes[1, idx].set_title(f'이중 슬릿: a = {a*1e6:.0f} um, d = {d*1e6:.0f} um')
    axes[1, idx].set_xlabel('theta (도)')
    axes[1, idx].set_ylabel('I / I_0')
    axes[1, idx].legend(fontsize=9)
    axes[1, idx].grid(True, alpha=0.3)

plt.suptitle('프라운호퍼 회절 = 슬릿 함수의 푸리에 변환', fontsize=14)
plt.tight_layout()
plt.savefig('fraunhofer_diffraction.png', dpi=150, bbox_inches='tight')
plt.show()
```

**이중 슬릿(double slit)** 의 경우, 투과 함수는 두 직사각형의 합이므로:

$$A(x) = \text{rect}\left(\frac{x - d/2}{a}\right) + \text{rect}\left(\frac{x + d/2}{a}\right)$$

푸리에 변환의 선형성과 이동 정리에 의해:

$$I(\theta) \sim \text{sinc}^2(\beta) \cdot \cos^2(\delta) = \text{(회절 엔벨로프)} \times \text{(간섭 무늬)}$$

### 6.3 신호 처리와 스펙트럼 분석

실제 물리학 실험에서 가장 널리 쓰이는 FFT 응용은 **스펙트럼 분석(spectral analysis)** 입니다.

```python
import numpy as np
import matplotlib.pyplot as plt

# 실제 응용: 복합 신호의 스펙트럼 분석

# 샘플링 설정
f_s = 1000  # 샘플링 주파수 (Hz)
T = 2.0     # 총 관측 시간 (s)
N = int(f_s * T)
t = np.arange(N) / f_s

# 복합 신호: 여러 주파수 성분 + 잡음
# 물리학 실험에서 흔히 볼 수 있는 상황
f1, A1 = 50, 1.0    # 50 Hz, 진폭 1.0 (주 신호)
f2, A2 = 120, 0.5   # 120 Hz, 진폭 0.5 (2차 고조파 근처)
f3, A3 = 300, 0.2   # 300 Hz, 진폭 0.2 (약한 성분)

signal = (A1 * np.sin(2 * np.pi * f1 * t) +
          A2 * np.sin(2 * np.pi * f2 * t) +
          A3 * np.sin(2 * np.pi * f3 * t))

# 백색 잡음 추가
np.random.seed(42)
noise_level = 0.8
signal_noisy = signal + noise_level * np.random.randn(N)

# FFT 계산
X = np.fft.fft(signal_noisy)
freqs = np.fft.fftfreq(N, 1/f_s)

# 파워 스펙트럼 밀도 (Power Spectral Density)
psd = (2.0 / N) * np.abs(X[:N//2])**2
freqs_pos = freqs[:N//2]

# 윈도잉 (Hanning window) 적용 후 스펙트럼
window = np.hanning(N)
signal_windowed = signal_noisy * window
X_windowed = np.fft.fft(signal_windowed)
# 윈도우 보정: 에너지 보존을 위해 정규화
window_correction = np.sum(window**2) / N
psd_windowed = (2.0 / (N * window_correction)) * np.abs(X_windowed[:N//2])**2

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 시간 영역 신호
axes[0, 0].plot(t[:200], signal_noisy[:200], 'gray', alpha=0.7, label='잡음 신호')
axes[0, 0].plot(t[:200], signal[:200], 'b-', linewidth=1.5, label='원본 신호')
axes[0, 0].set_title('시간 영역 신호 (처음 0.2초)')
axes[0, 0].set_xlabel('t (s)')
axes[0, 0].set_ylabel('진폭')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 진폭 스펙트럼
amplitude_spectrum = (2.0 / N) * np.abs(X[:N//2])
axes[0, 1].plot(freqs_pos, amplitude_spectrum, 'b-', linewidth=1)
axes[0, 1].set_title('진폭 스펙트럼 |X(f)|')
axes[0, 1].set_xlabel('주파수 (Hz)')
axes[0, 1].set_ylabel('진폭')
axes[0, 1].set_xlim(0, 400)
axes[0, 1].grid(True, alpha=0.3)
# 피크 주파수 표시
for f_peak, A_peak, label in [(f1, A1, '50 Hz'), (f2, A2, '120 Hz'), (f3, A3, '300 Hz')]:
    axes[0, 1].annotate(label, xy=(f_peak, A_peak), fontsize=10,
                        arrowprops=dict(arrowstyle='->', color='red'),
                        xytext=(f_peak + 20, A_peak + 0.1))

# 파워 스펙트럼 밀도 (선형 스케일)
axes[1, 0].plot(freqs_pos, psd, 'b-', linewidth=1, alpha=0.5, label='직사각형 윈도우')
axes[1, 0].plot(freqs_pos, psd_windowed, 'r-', linewidth=1.5, label='해닝 윈도우')
axes[1, 0].set_title('파워 스펙트럼 밀도 (PSD)')
axes[1, 0].set_xlabel('주파수 (Hz)')
axes[1, 0].set_ylabel('파워')
axes[1, 0].set_xlim(0, 400)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 파워 스펙트럼 밀도 (로그 스케일, dB)
psd_db = 10 * np.log10(psd_windowed + 1e-20)  # 0 방지
axes[1, 1].plot(freqs_pos, psd_db, 'r-', linewidth=1.5)
axes[1, 1].set_title('파워 스펙트럼 밀도 (dB 스케일)')
axes[1, 1].set_xlabel('주파수 (Hz)')
axes[1, 1].set_ylabel('파워 (dB)')
axes[1, 1].set_xlim(0, 400)
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('FFT를 이용한 스펙트럼 분석', fontsize=14)
plt.tight_layout()
plt.savefig('spectral_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
```

**스펙트럼 분석의 실무 요령**:

1. **윈도잉 (Windowing)**: 유한 데이터를 FFT할 때, 양 끝의 불연속에 의한 **스펙트럼 누설(spectral leakage)** 을 줄이기 위해 해닝(Hanning), 해밍(Hamming), 블랙먼(Blackman) 등의 윈도우 함수를 적용합니다.

2. **제로 패딩 (Zero-padding)**: 데이터 뒤에 0을 추가하여 FFT 크기를 키우면, 주파수 축의 **보간(interpolation)** 효과를 얻습니다 (주파수 해상도 자체는 변하지 않음).

3. **dB 스케일**: 파워 스펙트럼을 dB(데시벨)로 표시하면, 크기가 매우 다른 성분들을 함께 볼 수 있습니다: $P_{\text{dB}} = 10 \log_{10}(P)$.

---

## 연습 문제

### 문제 1: 기본 변환 계산

다음 함수의 푸리에 변환을 해석적으로 구하시오.

(a) $f(x) = e^{-3|x|}$

(b) $f(x) = x e^{-x^2}$

(c) $f(x) = \frac{1}{1 + x^2}$

> **힌트**: (a)는 양의 영역과 음의 영역으로 나누어 적분하세요. (b)는 가우시안 변환의 미분 성질을 활용하세요. (c)는 유수 정리(residue theorem)를 사용하거나, 변환 쌍 표를 참고하세요.

### 문제 2: 성질 활용

$f(x) = e^{-x^2}$의 푸리에 변환이 $F(\omega) = \sqrt{\pi} e^{-\omega^2/4}$임을 이용하여:

(a) $g(x) = e^{-(x-3)^2}$의 푸리에 변환을 구하시오. (시간 이동)

(b) $h(x) = e^{-4x^2}$의 푸리에 변환을 구하시오. (스케일링)

(c) $p(x) = x^2 e^{-x^2}$의 푸리에 변환을 구하시오. (주파수 미분)

### 문제 3: 컨볼루션

(a) $\text{rect}(x) * \text{rect}(x)$를 계산하시오 (직접 적분 또는 변환 이용). 결과가 삼각형 함수(triangle function)임을 보이시오.

(b) 열 전도 방정식에서, 초기 온도 분포가 $\delta(x)$인 경우의 해가 가우시안과 가우시안의 컨볼루션으로 표현됨을 보이시오.

### 문제 4: FFT 실습

```python
# 다음 코드를 완성하여 미지의 신호에서 주파수 성분을 찾으시오.

import numpy as np
import matplotlib.pyplot as plt

# 미지의 신호 생성
np.random.seed(2024)
f_s = 500  # 샘플링 주파수
t = np.arange(0, 2, 1/f_s)

# 3개의 숨겨진 주파수 성분 + 잡음
signal = (0.7 * np.sin(2*np.pi*___*t) +
          1.2 * np.sin(2*np.pi*___*t) +
          0.4 * np.cos(2*np.pi*___*t) +
          0.5 * np.random.randn(len(t)))

# TODO: FFT를 수행하고, 숨겨진 주파수를 찾으시오
# 1. np.fft.fft()로 FFT 수행
# 2. np.fft.fftfreq()로 주파수 축 계산
# 3. 진폭 스펙트럼을 그려서 피크 주파수 확인
```

### 문제 5: 물리학 응용

(a) 폭 $a = 50$ μm인 단일 슬릿에 파장 $\lambda = 632.8$ nm (He-Ne 레이저)의 빛을 통과시킬 때, 첫 번째 극소(first minimum)의 각도 $\theta$를 구하시오.

(b) 가우시안 파동 패킷 $\psi(x) = A e^{-x^2/(4\sigma^2)} e^{ik_0 x}$에 대해:
- 운동량 공간 파동함수 $\phi(k)$를 구하시오
- $\Delta x \cdot \Delta k$를 계산하고, 불확정성 원리를 만족함을 확인하시오

(c) 질량 $m$인 조화진동자(harmonic oscillator)의 바닥 상태(ground state)가 가우시안 형태인 이유를 불확정성 원리와 연결하여 설명하시오.

### 문제 6: 파르세발 정리

**파르세발 정리 (Parseval's theorem)**: 시간 영역과 주파수 영역에서의 에너지가 보존됨을 증명하시오.

$$\int_{-\infty}^{\infty} |f(x)|^2 \, dx = \frac{1}{2\pi} \int_{-\infty}^{\infty} |F(\omega)|^2 \, d\omega$$

> **힌트**: $f(x)$와 $f^*(x)$의 곱을 역 변환으로 표현하고, 적분 순서를 교환하세요.

---

## 참고 자료

### 교재

- **Mary L. Boas**, *Mathematical Methods in the Physical Sciences*, 3rd Edition
  - Chapter 7: Fourier Series and Transforms
  - Chapter 15: Integral Transforms (심화)
- **Arfken, Weber, Harris**, *Mathematical Methods for Physicists*, 7th Edition, Chapter 20
- **Riley, Hobson, Bence**, *Mathematical Methods for Physics and Engineering*, Chapter 13

### 관련 주제 심화

- **Bracewell, R.N.**, *The Fourier Transform and Its Applications*, McGraw-Hill
  - 푸리에 변환의 바이블이라 불리는 고전적 참고서
- **Oppenheim, A.S. & Willsky, A.S.**, *Signals and Systems*, Prentice Hall
  - 신호 처리 관점에서의 푸리에 변환
- **Goodman, J.W.**, *Introduction to Fourier Optics*, W.H. Freeman
  - 광학에서의 푸리에 변환 응용 (회절, 이미징)

### 온라인 자료

- **3Blue1Brown**: "But what is the Fourier Transform?" (시각적 직관)
- **MIT OCW 18.03**: Differential Equations (푸리에 급수/변환 강의)
- **NumPy FFT 공식 문서**: https://numpy.org/doc/stable/reference/routines.fft.html
- **SciPy FFT 공식 문서**: https://docs.scipy.org/doc/scipy/reference/fft.html

---

## 다음 레슨

[09. 1차·2차 상미분방정식 (ODE: First and Second Order)](09_ODE_First_Second_Order.md)에서는 1차 및 2차 상미분방정식의 풀이법, 초기값 문제, 물리학 응용을 다룹니다.
