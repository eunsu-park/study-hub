# 디지털 필터 기초

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 구조, 안정성, 위상 응답(Phase Response), 계산 비용 측면에서 FIR과 IIR 필터를 구분할 수 있습니다.
2. 차분 방정식(Difference Equation), Z 영역 전달 함수(Transfer Function), 주파수 응답을 사용하여 디지털 필터를 분석할 수 있습니다.
3. FIR 필터의 선형 위상(Linear Phase) 조건을 설명하고 위상에 민감한 응용에서의 중요성을 기술할 수 있습니다.
4. 통과대역 리플(Passband Ripple), 저지대역 감쇠(Stopband Attenuation), 전이 대역폭(Transition Width)을 포함한 필터 규격을 해석할 수 있습니다.
5. FIR 및 IIR 필터를 직접형(Direct Form), 종속형(Cascade), 격자형(Lattice) 구조로 구현할 수 있습니다.
6. 고정소수점 구현에서 계수 양자화(Coefficient Quantization), 오버플로우(Overflow), 한계 사이클(Limit Cycle)을 포함한 양자화 효과를 인식하고 평가할 수 있습니다.

---

## 개요

디지털 필터는 신호 처리의 핵심 도구로, 주파수 성분을 선택적으로 통과시키거나 제거함으로써 신호의 스펙트럼을 형성합니다. 이 레슨에서는 두 가지 기본 필터 유형인 FIR과 IIR의 표현 방식, 설계 트레이드오프, 구현 구조, 그리고 양자화 효과를 포함한 실용적인 고려 사항을 다룹니다. 이 기초를 이해하는 것은 구체적인 필터 설계 방법을 배우기 전에 반드시 필요합니다.

**선수 학습:** [07. Z-변환](07_Z_Transform.md)

---

## 1. 디지털 필터 유형: FIR과 IIR

### 1.1 FIR (유한 임펄스 응답, Finite Impulse Response)

FIR 필터는 유한 길이의 임펄스 응답을 가집니다. 출력은 현재와 과거의 입력에만 의존합니다:

$$y[n] = \sum_{k=0}^{M} b_k \, x[n-k]$$

**특성:**
- 비재귀적(Non-recursive) (피드백 없음)
- 항상 안정 ($z = 0$을 제외한 극점 없음)
- 정확한 선형 위상 구현 가능
- 날카로운 주파수 선택성을 위해 더 많은 계수 필요
- 전달 함수: $H(z) = \sum_{k=0}^{M} b_k z^{-k}$ (영점만 존재, 극점 없음)

### 1.2 IIR (무한 임펄스 응답, Infinite Impulse Response)

IIR 필터는 무한 길이의 임펄스 응답을 가집니다. 출력은 입력과 이전 출력에 의존합니다:

$$y[n] = \sum_{k=0}^{M} b_k \, x[n-k] - \sum_{k=1}^{N} a_k \, y[n-k]$$

**특성:**
- 재귀적(Recursive) (피드백 사용)
- 극점이 단위원 밖에 있으면 불안정해질 수 있음
- 일반적으로 정확한 선형 위상 구현 불가
- 더 효율적: 적은 계수로 날카로운 전이 대역 구현
- 전달 함수: $H(z) = \frac{\sum_{k=0}^{M} b_k z^{-k}}{1 + \sum_{k=1}^{N} a_k z^{-k}}$ (극점과 영점 모두 존재)

### 1.3 FIR vs. IIR 비교

| 속성 | FIR | IIR |
|------|-----|-----|
| 안정성 | 항상 안정 | 불안정 가능 |
| 선형 위상 | 구현 가능 | 일반적으로 불가 |
| 급격한 차단을 위한 차수 | 높음 (50-200+) | 낮음 (4-10) |
| 연산 비용 | 더 많은 곱셈-덧셈 | 더 적은 곱셈-덧셈 |
| 과도 응답 | 유한 (M 샘플) | 무한 (감쇠) |
| 설계 방법 | 윈도잉(Windowing), Parks-McClellan | Butterworth, Chebyshev, Elliptic |
| 아날로그 원형(prototype) | 불필요 | 자주 사용 |
| 양자화 민감도 | 낮음 | 더 높음 |
| 지연(Latency) | 더 높음 (긴 필터) | 더 낮음 |
| 군지연(Group delay) | 일정 (선형 위상) | 주파수 의존적 |

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def compare_fir_iir():
    """Compare FIR and IIR lowpass filters with similar specifications."""
    fs = 1000  # Hz
    f_cutoff = 100  # Hz

    # FIR: 51-tap lowpass using Hamming window
    fir_order = 50
    b_fir = signal.firwin(fir_order + 1, f_cutoff, fs=fs)
    a_fir = [1.0]

    # IIR: 4th-order Butterworth lowpass
    iir_order = 4
    b_iir, a_iir = signal.butter(iir_order, f_cutoff, fs=fs)

    # Frequency responses
    w_fir, H_fir = signal.freqz(b_fir, a_fir, worN=2048, fs=fs)
    w_iir, H_iir = signal.freqz(b_iir, a_iir, worN=2048, fs=fs)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Magnitude response
    axes[0, 0].plot(w_fir, 20 * np.log10(np.abs(H_fir) + 1e-12),
                    'b-', linewidth=1.5, label=f'FIR (order {fir_order})')
    axes[0, 0].plot(w_iir, 20 * np.log10(np.abs(H_iir) + 1e-12),
                    'r-', linewidth=1.5, label=f'IIR Butterworth (order {iir_order})')
    axes[0, 0].axvline(f_cutoff, color='green', linestyle='--', alpha=0.5,
                        label=f'Cutoff = {f_cutoff} Hz')
    axes[0, 0].set_title('Magnitude Response')
    axes[0, 0].set_xlabel('Frequency (Hz)')
    axes[0, 0].set_ylabel('Magnitude (dB)')
    axes[0, 0].set_ylim(-80, 5)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Phase response
    axes[0, 1].plot(w_fir, np.unwrap(np.angle(H_fir)) * 180 / np.pi,
                    'b-', linewidth=1.5, label='FIR')
    axes[0, 1].plot(w_iir, np.unwrap(np.angle(H_iir)) * 180 / np.pi,
                    'r-', linewidth=1.5, label='IIR')
    axes[0, 1].set_title('Phase Response')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Phase (degrees)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Group delay
    w_gd_fir, gd_fir = signal.group_delay((b_fir, a_fir), w=2048, fs=fs)
    w_gd_iir, gd_iir = signal.group_delay((b_iir, a_iir), w=2048, fs=fs)

    axes[1, 0].plot(w_gd_fir, gd_fir, 'b-', linewidth=1.5, label='FIR')
    axes[1, 0].plot(w_gd_iir, gd_iir, 'r-', linewidth=1.5, label='IIR')
    axes[1, 0].set_title('Group Delay')
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Delay (samples)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0, fs / 2)

    # Impulse response
    N_imp = 80
    imp = np.zeros(N_imp)
    imp[0] = 1.0

    y_fir = signal.lfilter(b_fir, a_fir, imp)
    y_iir = signal.lfilter(b_iir, a_iir, imp)

    axes[1, 1].stem(np.arange(N_imp), y_fir, linefmt='b-', markerfmt='b.',
                    basefmt='k-', label='FIR')
    axes[1, 1].stem(np.arange(N_imp), y_iir, linefmt='r-', markerfmt='r.',
                    basefmt='k-', label='IIR')
    axes[1, 1].set_title('Impulse Response')
    axes[1, 1].set_xlabel('n')
    axes[1, 1].set_ylabel('h[n]')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'FIR (order {fir_order}) vs. IIR Butterworth (order {iir_order})',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig('fir_vs_iir.png', dpi=150)
    plt.show()

    # Print coefficients
    print("FIR vs IIR Comparison")
    print("=" * 50)
    print(f"FIR: {fir_order + 1} coefficients (multiply-adds per sample)")
    print(f"IIR: {len(b_iir) + len(a_iir) - 1} coefficients "
          f"({len(b_iir)} numerator + {len(a_iir) - 1} denominator)")

compare_fir_iir()
```

---

## 2. 차분 방정식과 전달 함수

### 2.1 일반 차분 방정식

일반적인 선형 상수계수 차분 방정식(LCCDE, Linear Constant-Coefficient Difference Equation):

$$\sum_{k=0}^{N} a_k \, y[n-k] = \sum_{k=0}^{M} b_k \, x[n-k]$$

관습적으로 $a_0 = 1$로 설정하면:

$$y[n] = \sum_{k=0}^{M} b_k \, x[n-k] - \sum_{k=1}^{N} a_k \, y[n-k]$$

### 2.2 z-도메인에서의 전달 함수

Z-변환을 적용하면 (초기 조건 0 가정):

$$H(z) = \frac{Y(z)}{X(z)} = \frac{\sum_{k=0}^{M} b_k z^{-k}}{1 + \sum_{k=1}^{N} a_k z^{-k}} = \frac{B(z)}{A(z)}$$

인수분해 형태:

$$H(z) = \frac{b_0}{a_0} \cdot \frac{\prod_{k=1}^{M}(1 - z_k z^{-1})}{\prod_{k=1}^{N}(1 - p_k z^{-1})}$$

여기서 $z_k$는 영점(zero), $p_k$는 극점(pole)입니다.

### 2.3 주파수 응답

단위원 위에서 $H(z)$를 평가하면 주파수 응답을 얻습니다:

$$H(e^{j\omega}) = |H(e^{j\omega})| \, e^{j\phi(\omega)}$$

- $|H(e^{j\omega})|$: **크기 응답(Magnitude response)** (주파수 $\omega$에서의 진폭 이득)
- $\phi(\omega) = \angle H(e^{j\omega})$: **위상 응답(Phase response)** (주파수 $\omega$에서의 위상 이동)

```python
def difference_equation_to_transfer_function():
    """Demonstrate the relationship between difference equation,
    transfer function, and frequency response."""
    # Example: y[n] = 0.5(x[n] + x[n-1]) - Simple moving average
    # This is a 2-tap FIR (order 1)
    b_ma = [0.5, 0.5]
    a_ma = [1.0]

    # Example: y[n] = 0.9*y[n-1] + 0.1*x[n] - First-order IIR lowpass
    b_iir = [0.1]
    a_iir = [1.0, -0.9]

    systems = [
        ('2-tap Moving Average (FIR)', b_ma, a_ma),
        ('First-order IIR Lowpass', b_iir, a_iir),
    ]

    fig, axes = plt.subplots(len(systems), 3, figsize=(16, 8))

    for i, (name, b, a) in enumerate(systems):
        # Impulse response
        N = 30
        imp = np.zeros(N)
        imp[0] = 1.0
        h = signal.lfilter(b, a, imp)

        axes[i, 0].stem(np.arange(N), h, linefmt='b-', markerfmt='bo',
                        basefmt='k-')
        axes[i, 0].set_title(f'{name}\nImpulse Response')
        axes[i, 0].set_xlabel('n')
        axes[i, 0].set_ylabel('h[n]')
        axes[i, 0].grid(True, alpha=0.3)

        # Magnitude response
        w, H = signal.freqz(b, a, worN=1024)
        axes[i, 1].plot(w / np.pi, 20 * np.log10(np.abs(H) + 1e-12),
                        'b-', linewidth=2)
        axes[i, 1].set_title('Magnitude Response')
        axes[i, 1].set_xlabel(r'$\omega / \pi$')
        axes[i, 1].set_ylabel('dB')
        axes[i, 1].set_ylim(-40, 5)
        axes[i, 1].grid(True, alpha=0.3)

        # Phase response
        axes[i, 2].plot(w / np.pi, np.unwrap(np.angle(H)) * 180 / np.pi,
                        'r-', linewidth=2)
        axes[i, 2].set_title('Phase Response')
        axes[i, 2].set_xlabel(r'$\omega / \pi$')
        axes[i, 2].set_ylabel('Degrees')
        axes[i, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('diff_eq_to_tf.png', dpi=150)
    plt.show()

difference_equation_to_transfer_function()
```

---

## 3. FIR 필터 특성

### 3.1 FIR 전달 함수

$$H(z) = \sum_{k=0}^{M} b_k z^{-k} = b_0 + b_1 z^{-1} + b_2 z^{-2} + \cdots + b_M z^{-M}$$

- 원점 $z = 0$에 $M$개의 극점과 $M$개의 영점
- 차수 $M$, 길이 $M + 1$
- 항상 안정 (모든 극점이 원점에 위치)

### 3.2 $z^{-1}$의 다항식으로서의 FIR

FIR 필터는 단순히 다항식 평가기(polynomial evaluator)입니다. 각 계수 $b_k$는 임펄스 응답 값 $h[k]$와 동일합니다:

$$h[n] = b_n, \quad n = 0, 1, \ldots, M$$

### 3.3 일반적인 FIR 필터 유형

```python
def common_fir_filters():
    """Demonstrate common FIR filter building blocks."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # 1. Moving average (lowpass)
    M = 10
    b_ma = np.ones(M + 1) / (M + 1)
    w, H_ma = signal.freqz(b_ma, [1.0], worN=2048)

    axes[0, 0].stem(np.arange(M + 1), b_ma, linefmt='b-', markerfmt='bo',
                    basefmt='k-')
    axes[0, 0].set_title(f'Moving Average (M={M}): Coefficients')
    axes[0, 0].set_xlabel('n')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(w / np.pi, 20 * np.log10(np.abs(H_ma) + 1e-12),
                    'b-', linewidth=2)
    axes[0, 1].set_title('Moving Average: Magnitude Response')
    axes[0, 1].set_xlabel(r'$\omega / \pi$')
    axes[0, 1].set_ylabel('dB')
    axes[0, 1].set_ylim(-60, 5)
    axes[0, 1].grid(True, alpha=0.3)

    # 2. Differentiator
    b_diff = [1, -1]
    w, H_diff = signal.freqz(b_diff, [1.0], worN=2048)

    axes[1, 0].stem(np.arange(len(b_diff)), b_diff, linefmt='r-',
                    markerfmt='ro', basefmt='k-')
    axes[1, 0].set_title('First Difference (Differentiator): Coefficients')
    axes[1, 0].set_xlabel('n')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(w / np.pi, np.abs(H_diff), 'r-', linewidth=2)
    axes[1, 1].set_title('Differentiator: Magnitude Response')
    axes[1, 1].set_xlabel(r'$\omega / \pi$')
    axes[1, 1].set_ylabel('|H|')
    axes[1, 1].grid(True, alpha=0.3)

    # 3. Comb filter
    D = 8  # Delay
    b_comb = np.zeros(D + 1)
    b_comb[0] = 1
    b_comb[D] = -1
    w, H_comb = signal.freqz(b_comb, [1.0], worN=2048)

    axes[2, 0].stem(np.arange(len(b_comb)), b_comb, linefmt='g-',
                    markerfmt='go', basefmt='k-')
    axes[2, 0].set_title(f'Comb Filter (D={D}): Coefficients')
    axes[2, 0].set_xlabel('n')
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].plot(w / np.pi, 20 * np.log10(np.abs(H_comb) + 1e-12),
                    'g-', linewidth=2)
    axes[2, 1].set_title('Comb Filter: Magnitude Response')
    axes[2, 1].set_xlabel(r'$\omega / \pi$')
    axes[2, 1].set_ylabel('dB')
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('common_fir.png', dpi=150)
    plt.show()

common_fir_filters()
```

---

## 4. IIR 필터 특성

### 4.1 IIR 전달 함수

$$H(z) = \frac{B(z)}{A(z)} = \frac{\sum_{k=0}^{M} b_k z^{-k}}{1 + \sum_{k=1}^{N} a_k z^{-k}}$$

- $A(z)$에서 극점과 $B(z)$에서 영점 모두 존재
- 필터 차수 $= \max(M, N)$, 일반적으로 $N$ (분모 차수)
- 안정성을 위해 모든 극점이 단위원 내에 있어야 함

### 4.2 IIR 필터 계열

고전적인 IIR 필터 설계는 아날로그 원형(analog prototype)에서 출발합니다:

| 계열 | 통과대역 | 저지대역 | 전이 대역 | 위상 |
|------|---------|---------|---------|------|
| **Butterworth** | 최대 평탄 | 단조 | 가장 넓음 | 가장 부드러움 |
| **Chebyshev Type I** | 등리플(Equiripple) | 단조 | 더 좁음 | 더 비선형 |
| **Chebyshev Type II** | 단조 | 등리플 | 더 좁음 | 더 비선형 |
| **Elliptic (Cauer)** | 등리플 | 등리플 | 가장 좁음 | 가장 비선형 |
| **Bessel** | 거의 평탄 | 낮음 | 가장 넓음 | 가장 선형 |

```python
def compare_iir_families():
    """Compare the four main IIR filter families."""
    fs = 1000
    f_cutoff = 100
    order = 5

    # Design filters
    filters = {
        'Butterworth': signal.butter(order, f_cutoff, fs=fs),
        'Chebyshev I (1dB ripple)': signal.cheby1(order, 1, f_cutoff, fs=fs),
        'Chebyshev II (40dB atten)': signal.cheby2(order, 40, f_cutoff, fs=fs),
        'Elliptic (1dB/40dB)': signal.ellip(order, 1, 40, f_cutoff, fs=fs),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = ['blue', 'red', 'green', 'orange']

    # Magnitude response
    ax = axes[0, 0]
    for (name, (b, a)), color in zip(filters.items(), colors):
        w, H = signal.freqz(b, a, worN=4096, fs=fs)
        ax.plot(w, 20 * np.log10(np.abs(H) + 1e-12), color=color,
                linewidth=1.5, label=name)
    ax.axvline(f_cutoff, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f'Magnitude Response (Order {order})')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_ylim(-80, 5)
    ax.set_xlim(0, 250)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Passband detail
    ax = axes[0, 1]
    for (name, (b, a)), color in zip(filters.items(), colors):
        w, H = signal.freqz(b, a, worN=4096, fs=fs)
        ax.plot(w, 20 * np.log10(np.abs(H) + 1e-12), color=color,
                linewidth=1.5, label=name)
    ax.axvline(f_cutoff, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Passband Detail')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_ylim(-3, 1)
    ax.set_xlim(0, 120)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Group delay
    ax = axes[1, 0]
    for (name, (b, a)), color in zip(filters.items(), colors):
        w, gd = signal.group_delay((b, a), w=4096, fs=fs)
        ax.plot(w, gd, color=color, linewidth=1.5, label=name)
    ax.set_title('Group Delay')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Delay (samples)')
    ax.set_xlim(0, 200)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Pole-zero plot for Butterworth
    ax = axes[1, 1]
    b_bw, a_bw = filters['Butterworth']
    zeros = np.roots(b_bw)
    poles = np.roots(a_bw)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=0.5)
    ax.plot(np.real(zeros), np.imag(zeros), 'bo', markersize=8, label='Zeros')
    ax.plot(np.real(poles), np.imag(poles), 'rx', markersize=10,
            markeredgewidth=2, label='Poles')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title(f'Butterworth Order {order}: Pole-Zero Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('iir_families.png', dpi=150)
    plt.show()

compare_iir_families()
```

### 4.3 IIR 필터의 잠재적 불안정성

```python
def demonstrate_iir_instability():
    """Show how IIR filters can become unstable."""
    # Stable filter: pole at 0.95
    b_stable = [1.0]
    a_stable = [1.0, -0.95]

    # Unstable filter: pole at 1.05
    b_unstable = [1.0]
    a_unstable = [1.0, -1.05]

    N = 50
    imp = np.zeros(N)
    imp[0] = 1.0

    h_stable = signal.lfilter(b_stable, a_stable, imp)
    h_unstable = signal.lfilter(b_unstable, a_unstable, imp)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].stem(np.arange(N), h_stable, linefmt='g-', markerfmt='go',
                 basefmt='k-')
    axes[0].set_title('Stable IIR (pole = 0.95)')
    axes[0].set_xlabel('n')
    axes[0].set_ylabel('h[n]')
    axes[0].grid(True, alpha=0.3)

    axes[1].stem(np.arange(N), h_unstable, linefmt='r-', markerfmt='ro',
                 basefmt='k-')
    axes[1].set_title('UNSTABLE IIR (pole = 1.05)')
    axes[1].set_xlabel('n')
    axes[1].set_ylabel('h[n]')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('iir_instability.png', dpi=150)
    plt.show()

    print("IIR Stability Demonstration")
    print("=" * 50)
    print(f"Stable: h[49] = {h_stable[49]:.6e}")
    print(f"Unstable: h[49] = {h_unstable[49]:.6e}")

demonstrate_iir_instability()
```

---

## 5. 주파수 응답: 크기와 위상

### 5.1 크기 응답(Magnitude Response)

$$|H(e^{j\omega})| = \sqrt{\text{Re}[H(e^{j\omega})]^2 + \text{Im}[H(e^{j\omega})]^2}$$

데시벨로 표현: $|H|_{\text{dB}} = 20 \log_{10}|H(e^{j\omega})|$

### 5.2 위상 응답(Phase Response)

$$\phi(\omega) = \angle H(e^{j\omega}) = \arctan\!\left(\frac{\text{Im}[H]}{\text{Re}[H]}\right)$$

### 5.3 군지연(Group Delay)

**군지연**은 위상 응답의 음의 미분입니다:

$$\tau_g(\omega) = -\frac{d\phi(\omega)}{d\omega}$$

군지연은 주파수 $\omega$를 중심으로 하는 협대역 신호의 포락선이 경험하는 지연을 나타냅니다. 선형 위상 필터의 경우 군지연은 일정합니다.

### 5.4 위상 지연(Phase Delay)

**위상 지연**은 다음과 같습니다:

$$\tau_p(\omega) = -\frac{\phi(\omega)}{\omega}$$

위상 지연은 주파수 $\omega$의 순수한 정현파가 경험하는 지연을 나타냅니다.

선형 위상의 경우: $\phi(\omega) = -\omega \tau_0$이므로 $\tau_g = \tau_p = \tau_0$ (일정).

```python
def phase_and_group_delay():
    """Demonstrate phase response, group delay, and phase delay."""
    fs = 1000

    # FIR with linear phase (symmetric coefficients)
    N_fir = 31
    b_fir = signal.firwin(N_fir, 200, fs=fs)
    a_fir = [1.0]

    # IIR (nonlinear phase)
    b_iir, a_iir = signal.butter(6, 200, fs=fs)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    titles = ['FIR (Linear Phase)', 'IIR Butterworth (Nonlinear Phase)']
    systems = [(b_fir, a_fir), (b_iir, a_iir)]

    for col, ((b, a), title) in enumerate(zip(systems, titles)):
        w, H = signal.freqz(b, a, worN=2048, fs=fs)

        # Magnitude
        axes[0, col].plot(w, 20 * np.log10(np.abs(H) + 1e-12),
                          'b-', linewidth=2)
        axes[0, col].set_title(f'{title}\nMagnitude Response')
        axes[0, col].set_xlabel('Frequency (Hz)')
        axes[0, col].set_ylabel('dB')
        axes[0, col].set_ylim(-80, 5)
        axes[0, col].grid(True, alpha=0.3)

        # Phase (unwrapped)
        phase = np.unwrap(np.angle(H))
        axes[1, col].plot(w, phase * 180 / np.pi, 'r-', linewidth=2)
        axes[1, col].set_title('Phase Response')
        axes[1, col].set_xlabel('Frequency (Hz)')
        axes[1, col].set_ylabel('Phase (degrees)')
        axes[1, col].grid(True, alpha=0.3)

        # Group delay
        w_gd, gd = signal.group_delay((b, a), w=2048, fs=fs)
        axes[2, col].plot(w_gd, gd, 'g-', linewidth=2)
        axes[2, col].set_title('Group Delay')
        axes[2, col].set_xlabel('Frequency (Hz)')
        axes[2, col].set_ylabel('Delay (samples)')
        axes[2, col].set_xlim(0, fs / 2)
        axes[2, col].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('phase_group_delay.png', dpi=150)
    plt.show()

phase_and_group_delay()
```

---

## 6. 선형 위상 FIR 필터

### 6.1 선형 위상이 중요한 이유

선형 위상 필터는 다음과 같은 위상 응답을 가집니다:

$$\phi(\omega) = -\alpha \omega + \beta$$

여기서 $\alpha$는 일정한 군지연이고 $\beta$는 0 또는 $\pi/2$입니다.

**중요성:**
- 모든 주파수 성분이 동일한 시간만큼 지연됨
- 신호 파형의 위상 왜곡 없음
- 오디오, 통신, 영상 처리 등의 응용에 필수적
- 과도(transient) 신호의 형태 보존

### 6.2 대칭 조건

길이 $N = M+1$인 FIR 필터 $h[n]$이 선형 위상을 가지려면 계수가 다음 두 대칭 조건 중 하나를 만족해야 합니다:

**대칭 (Type I 및 II):**
$$h[n] = h[M-n], \quad n = 0, 1, \ldots, M$$

**반대칭(Anti-symmetric) (Type III 및 IV):**
$$h[n] = -h[M-n], \quad n = 0, 1, \ldots, M$$

### 6.3 선형 위상 FIR 필터의 네 가지 유형

| 유형 | 대칭 | 길이 ($M+1$) | 위상 ($\beta$) | 적합한 용도 |
|------|------|-------------|-------------|----------|
| **I** | 대칭 | 홀수 | 0 | LP, HP, BP, BS |
| **II** | 대칭 | 짝수 | 0 | LP, BP만 |
| **III** | 반대칭 | 홀수 | $\pi/2$ | BP, 미분기(differentiator) |
| **IV** | 반대칭 | 짝수 | $\pi/2$ | HP, BP, 미분기, 힐버트(Hilbert) |

**제약 조건:**
- Type II: $H(e^{j\pi}) = 0$ (고역통과 불가)
- Type III: $H(e^{j0}) = 0$ 및 $H(e^{j\pi}) = 0$ (저역통과 및 고역통과 불가)
- Type IV: $H(e^{j0}) = 0$ (저역통과 불가)

```python
def linear_phase_types():
    """Demonstrate all four types of linear phase FIR filters."""
    fig, axes = plt.subplots(4, 3, figsize=(16, 16))

    # Type I: Symmetric, Odd length (lowpass)
    M = 20  # Order (even -> odd length M+1=21)
    h1 = signal.firwin(M + 1, 0.4)  # Symmetric by construction
    assert len(h1) % 2 == 1  # Odd length

    # Type II: Symmetric, Even length (lowpass)
    M2 = 19  # Order (odd -> even length M+1=20)
    h2 = signal.firwin(M2 + 1, 0.4)

    # Type III: Anti-symmetric, Odd length (bandpass/differentiator)
    M3 = 20
    h3 = signal.firwin(M3 + 1, [0.2, 0.8], pass_zero=False)
    # Make it anti-symmetric: h[n] = -h[M-n]
    h3_anti = (h3 - h3[::-1]) / 2
    # But more naturally, use remez for differentiator
    h3_diff = signal.remez(M3 + 1, [0.05, 0.95], [1], type='differentiator')

    # Type IV: Anti-symmetric, Even length (Hilbert transformer)
    M4 = 19
    h4 = signal.remez(M4 + 1, [0.05, 0.95], [1], type='hilbert')

    types_data = [
        ('Type I (Symmetric, Odd)', h1, 'blue'),
        ('Type II (Symmetric, Even)', h2, 'red'),
        ('Type III (Anti-sym, Odd)', h3_diff, 'green'),
        ('Type IV (Anti-sym, Even)', h4, 'orange'),
    ]

    for row, (name, h, color) in enumerate(types_data):
        N = len(h)
        n = np.arange(N)

        # Coefficients
        axes[row, 0].stem(n, h, linefmt=f'{color[0]}-', markerfmt=f'{color[0]}o',
                          basefmt='k-')
        axes[row, 0].set_title(f'{name}\nCoefficients (N={N})')
        axes[row, 0].set_xlabel('n')
        axes[row, 0].grid(True, alpha=0.3)

        # Check symmetry
        is_symmetric = np.allclose(h, h[::-1], atol=1e-10)
        is_antisymmetric = np.allclose(h, -h[::-1], atol=1e-10)
        sym_text = 'Symmetric' if is_symmetric else \
                   'Anti-symmetric' if is_antisymmetric else 'Neither'
        axes[row, 0].text(0.95, 0.95, sym_text, transform=axes[row, 0].transAxes,
                          ha='right', va='top', fontsize=9,
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Magnitude response
        w, H_freq = signal.freqz(h, [1.0], worN=2048)
        axes[row, 1].plot(w / np.pi, 20 * np.log10(np.abs(H_freq) + 1e-12),
                          color=color, linewidth=2)
        axes[row, 1].set_title('Magnitude Response')
        axes[row, 1].set_xlabel(r'$\omega / \pi$')
        axes[row, 1].set_ylabel('dB')
        axes[row, 1].set_ylim(-80, 10)
        axes[row, 1].grid(True, alpha=0.3)

        # Group delay
        w_gd, gd = signal.group_delay((h, [1.0]), w=2048)
        axes[row, 2].plot(w_gd / np.pi, gd, color=color, linewidth=2)
        axes[row, 2].set_title(f'Group Delay (expected: {(N-1)/2:.1f})')
        axes[row, 2].set_xlabel(r'$\omega / \pi$')
        axes[row, 2].set_ylabel('Samples')
        axes[row, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('linear_phase_types.png', dpi=150)
    plt.show()

linear_phase_types()
```

### 6.4 증명: 대칭 FIR 필터의 선형 위상

Type I 필터 ($M$ 짝수, 대칭: $h[n] = h[M-n]$)의 경우:

$$H(e^{j\omega}) = e^{-j\omega M/2} \underbrace{\left[ h[M/2] + 2\sum_{k=1}^{M/2} h[M/2-k] \cos(k\omega) \right]}_{\tilde{H}(\omega) \text{ (실수값)}}$$

진폭 응답 $\tilde{H}(\omega)$는 순수 실수이며, 위상은 정확히 $-\omega M/2$ (선형)이고, $\tilde{H}(\omega) < 0$일 때 $\pi$만큼의 추가 이동이 발생합니다.

---

## 7. 필터 규격(Filter Specifications)

### 7.1 표준 규격

일반적인 필터는 다음과 같이 규격화됩니다:

```
    |H(e^jw)|
    ↑
1+δp ─ ─ ─ ─┐
    │        │  Passband ripple
1   │────────│
1-δp ─ ─ ─ ─│─ ─ ─ ─ ─ ─ ─ ─ ─
    │        │\
    │        │ \ Transition
    │        │  \  band
    │        │   \
 δs ─ ─ ─ ─ ─ ─ ─\─ ─ ─ ─ ─ ─
    │              │~~~~~~~~~~~
  0 ├──────┬──────┬──────┬────→ ω
    0      ωp    ωs             π
```

| 파라미터 | 기호 | 정의 |
|---------|------|------|
| 통과대역 엣지(Passband edge) | $\omega_p$ | 통과대역이 끝나는 주파수 |
| 저지대역 엣지(Stopband edge) | $\omega_s$ | 저지대역이 시작하는 주파수 |
| 통과대역 리플(Passband ripple) | $\delta_p$ | 통과대역에서 1로부터 최대 편차 |
| 저지대역 감쇠(Stopband attenuation) | $\delta_s$ | 저지대역에서의 최대 레벨 |
| 전이 대역폭(Transition width) | $\Delta\omega = \omega_s - \omega_p$ | 전이 대역의 폭 |

### 7.2 데시벨 변환

$$\text{통과대역 리플 (dB)} = -20 \log_{10}(1 - \delta_p)$$

$$\text{저지대역 감쇠 (dB)} = -20 \log_{10}(\delta_s)$$

| 규격 | 선형 | dB |
|------|------|----|
| 1% 통과대역 리플 | $\delta_p = 0.01$ | 0.087 dB |
| 0.1 dB 리플 | $\delta_p \approx 0.0115$ | 0.1 dB |
| 1 dB 리플 | $\delta_p \approx 0.109$ | 1 dB |
| 40 dB 저지대역 | $\delta_s = 0.01$ | 40 dB |
| 60 dB 저지대역 | $\delta_s = 0.001$ | 60 dB |
| 80 dB 저지대역 | $\delta_s = 0.0001$ | 80 dB |

### 7.3 필터 차수 추정

**FIR (Kaiser 공식):**

$$M \approx \frac{-20\log_{10}(\sqrt{\delta_p \delta_s}) - 13}{14.6 \cdot \Delta f / f_s}$$

**IIR (Butterworth):**

$$N \geq \frac{\log(\delta_p / \delta_s)}{2\log(\omega_p / \omega_s)}$$

```python
def filter_specification_demo():
    """Demonstrate filter specifications with a practical design."""
    fs = 8000  # Hz

    # Specifications
    f_pass = 1000   # Passband edge (Hz)
    f_stop = 1500   # Stopband edge (Hz)
    delta_p = 0.01  # Passband ripple (linear)
    delta_s = 0.001 # Stopband attenuation (linear)

    # Convert to dB
    rp_db = -20 * np.log10(1 - delta_p)
    rs_db = -20 * np.log10(delta_s)

    print("Filter Specifications")
    print("=" * 50)
    print(f"Sampling rate: {fs} Hz")
    print(f"Passband edge: {f_pass} Hz")
    print(f"Stopband edge: {f_stop} Hz")
    print(f"Transition width: {f_stop - f_pass} Hz")
    print(f"Passband ripple: {delta_p} ({rp_db:.3f} dB)")
    print(f"Stopband attenuation: {delta_s} ({rs_db:.1f} dB)")

    # Estimate FIR order (Kaiser)
    delta_f = (f_stop - f_pass) / fs
    M_kaiser = int(np.ceil(
        (-20 * np.log10(np.sqrt(delta_p * delta_s)) - 13) / (14.6 * delta_f)
    ))
    print(f"\nEstimated FIR order (Kaiser): {M_kaiser}")

    # Estimate IIR Butterworth order
    N_butter, Wn = signal.buttord(f_pass, f_stop, rp_db, rs_db, fs=fs)
    print(f"Estimated Butterworth order: {N_butter}")

    # Design both filters
    b_fir = signal.firwin(M_kaiser + 1, (f_pass + f_stop) / 2, fs=fs)

    b_iir, a_iir = signal.butter(N_butter, Wn, fs=fs)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    w_fir, H_fir = signal.freqz(b_fir, [1.0], worN=4096, fs=fs)
    w_iir, H_iir = signal.freqz(b_iir, a_iir, worN=4096, fs=fs)

    ax = axes[0]
    ax.plot(w_fir, 20 * np.log10(np.abs(H_fir) + 1e-12), 'b-',
            linewidth=1.5, label=f'FIR (order {M_kaiser})')
    ax.plot(w_iir, 20 * np.log10(np.abs(H_iir) + 1e-12), 'r-',
            linewidth=1.5, label=f'Butterworth (order {N_butter})')

    # Specification boundaries
    ax.axhline(-rp_db, color='green', linestyle=':', alpha=0.7,
               label=f'Passband: -{rp_db:.3f} dB')
    ax.axhline(-rs_db, color='orange', linestyle=':', alpha=0.7,
               label=f'Stopband: -{rs_db:.1f} dB')
    ax.axvline(f_pass, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(f_stop, color='gray', linestyle='--', alpha=0.5)
    ax.axvspan(0, f_pass, alpha=0.05, color='green')
    ax.axvspan(f_stop, fs / 2, alpha=0.05, color='red')

    ax.set_title('Filter Design Meeting Specifications')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_ylim(-80, 5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Passband detail
    ax = axes[1]
    ax.plot(w_fir, 20 * np.log10(np.abs(H_fir) + 1e-12), 'b-',
            linewidth=1.5, label=f'FIR (order {M_kaiser})')
    ax.plot(w_iir, 20 * np.log10(np.abs(H_iir) + 1e-12), 'r-',
            linewidth=1.5, label=f'Butterworth (order {N_butter})')
    ax.axhline(-rp_db, color='green', linestyle=':', alpha=0.7)
    ax.set_title('Passband Detail')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_xlim(0, f_stop * 1.2)
    ax.set_ylim(-3, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('filter_specs.png', dpi=150)
    plt.show()

filter_specification_demo()
```

---

## 8. 필터 구조

### 8.1 직접형 I (Direct Form I)

일반 차분 방정식의 가장 직접적인 구현:

$$y[n] = \sum_{k=0}^{M} b_k x[n-k] - \sum_{k=1}^{N} a_k y[n-k]$$

**신호 흐름:**
```
x[n] ──→[b0]──→(+)──→(+)──→ y[n]
  │              ↑      ↑
  ├→[z⁻¹]→[b1]──┘      │
  │                     │
  ├→[z⁻¹]→[b2]──→(+)   │
  ⋮              ↑      │
                       [-a1]←[z⁻¹]←── y[n]
                        │
                       [-a2]←[z⁻¹]
                        ⋮
```

**메모리 요구량:** $M + N$개의 지연 소자.

### 8.2 직접형 II (정준형, Canonical Form)

분자와 분모 간에 지연 소자를 공유하여 메모리를 줄입니다:

$$w[n] = x[n] - \sum_{k=1}^{N} a_k w[n-k]$$
$$y[n] = \sum_{k=0}^{M} b_k w[n-k]$$

**메모리 요구량:** $\max(M, N)$개의 지연 소자 (정준형: 최소 메모리).

```
x[n] ──→(+)──→ w[n] ──→[b0]──→(+)──→ y[n]
          ↑      │               ↑
          │      └→[z⁻¹]→w[n-1]─┤
          │      │    │          │
          │      │   [b1]───────┘
          │      │
         [-a1]───┤
          ↑      └→[z⁻¹]→w[n-2]─┐
          │           │          │
         [-a2]────────┘    [b2]──┘
```

### 8.3 전치 직접형 II (Transposed Direct Form II)

직접형 II의 신호 흐름 그래프를 전치(신호 흐름 반전, 분기점과 합산 접점 교환)하여 얻습니다:

$$v_1[n] = b_M x[n] - a_N y[n]$$
$$v_k[n] = v_{k-1}[n-1] + b_{M-k} x[n] - a_{N-k} y[n], \quad k = 2, \ldots, N$$
$$y[n] = v_N[n-1] + b_0 x[n]$$

이 구조는 중간 결과의 동적 범위가 더 작기 때문에 부동소수점 구현에서 수치적으로 선호됩니다.

### 8.4 종속형 (2차 섹션, Second-Order Sections) 형식

$H(z)$를 2차 섹션(바이쿼드, biquad)으로 인수분해합니다:

$$H(z) = G \prod_{k=1}^{L} \frac{b_{0k} + b_{1k}z^{-1} + b_{2k}z^{-2}}{1 + a_{1k}z^{-1} + a_{2k}z^{-2}}$$

여기서 $L = \lceil N/2 \rceil$.

**장점:**
- 각 섹션이 단순한 2차 필터(바이쿼드)
- 계수 양자화에 훨씬 덜 민감
- 개별 섹션 튜닝 용이
- 오디오 처리의 표준

```python
def demonstrate_filter_structures():
    """Implement and compare different filter structures."""
    # Design a 6th-order Butterworth lowpass
    order = 6
    fs = 8000
    fc = 1000
    b, a = signal.butter(order, fc, fs=fs)

    # Direct Form implementation
    def direct_form_1(b, a, x):
        """Direct Form I implementation."""
        M = len(b) - 1
        N_a = len(a) - 1
        y = np.zeros(len(x))
        x_buf = np.zeros(M + 1)
        y_buf = np.zeros(N_a + 1)

        for n in range(len(x)):
            # Shift buffers
            x_buf[1:] = x_buf[:-1]
            x_buf[0] = x[n]

            # FIR part
            y[n] = np.dot(b, x_buf)

            # IIR part
            if N_a > 0:
                y[n] -= np.dot(a[1:], y_buf[:N_a])

            # Update output buffer
            y_buf[1:] = y_buf[:-1]
            y_buf[0] = y[n]

        return y

    def direct_form_2(b, a, x):
        """Direct Form II (canonical) implementation."""
        M = len(b) - 1
        N_a = len(a) - 1
        K = max(M, N_a)
        y = np.zeros(len(x))
        w = np.zeros(K + 1)

        for n in range(len(x)):
            # Compute w[n]
            w[0] = x[n]
            for k in range(1, N_a + 1):
                w[0] -= a[k] * w[k]

            # Compute y[n]
            y[n] = 0
            for k in range(M + 1):
                y[n] += b[k] * w[k]

            # Shift w
            w[1:] = w[:-1]

        return y

    # Test signal
    N = 200
    t = np.arange(N) / fs
    x = np.sin(2 * np.pi * 500 * t) + 0.5 * np.sin(2 * np.pi * 2000 * t)

    # Compare implementations
    y_scipy = signal.lfilter(b, a, x)
    y_df1 = direct_form_1(b, a, x)
    y_df2 = direct_form_2(b, a, x)

    # Second-order sections (cascade form)
    sos = signal.butter(order, fc, fs=fs, output='sos')
    y_sos = signal.sosfilt(sos, x)

    print("Filter Structure Comparison")
    print("=" * 50)
    print(f"Filter: {order}th-order Butterworth, fc={fc} Hz, fs={fs} Hz")
    print(f"Coefficients b: {len(b)}, a: {len(a)}")
    print(f"\nMax error vs. scipy.lfilter:")
    print(f"  Direct Form I:  {np.max(np.abs(y_df1 - y_scipy)):.2e}")
    print(f"  Direct Form II: {np.max(np.abs(y_df2 - y_scipy)):.2e}")
    print(f"  SOS (cascade):  {np.max(np.abs(y_sos - y_scipy)):.2e}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(t * 1000, x, 'b-', alpha=0.5, label='Input')
    axes[0].plot(t * 1000, y_scipy, 'r-', linewidth=2, label='Filtered (scipy)')
    axes[0].set_title('Input and Filtered Signal')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t * 1000, y_df1 - y_scipy, 'b-', label='DF1 error')
    axes[1].plot(t * 1000, y_df2 - y_scipy, 'r-', label='DF2 error')
    axes[1].plot(t * 1000, y_sos - y_scipy, 'g-', label='SOS error')
    axes[1].set_title('Error vs. scipy.lfilter (numerical precision)')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Error')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('filter_structures.png', dpi=150)
    plt.show()

    # Print SOS sections
    print(f"\nSecond-Order Sections (SOS):")
    print(f"Number of sections: {len(sos)}")
    for i, section in enumerate(sos):
        print(f"  Section {i+1}: b={section[:3]}, a={section[3:]}")

demonstrate_filter_structures()
```

### 8.5 병렬형(Parallel Form)

$H(z)$를 2차 섹션의 합으로 분해합니다:

$$H(z) = c_0 + \sum_{k=1}^{L} \frac{b_{0k} + b_{1k}z^{-1}}{1 + a_{1k}z^{-1} + a_{2k}z^{-2}}$$

$H(z)$의 부분 분수 전개(partial fraction expansion)로 얻습니다.

### 8.6 격자형(Lattice) 구조

FIR 필터의 격자형 구조는 **반사 계수(reflection coefficients)** $k_m$을 사용합니다:

$$f_m[n] = f_{m-1}[n] + k_m \, g_{m-1}[n-1]$$
$$g_m[n] = k_m \, f_{m-1}[n] + g_{m-1}[n-1]$$

$f_0[n] = g_0[n] = x[n]$에서 시작.

**장점:**
- 모든 $m$에 대해 $|k_m| < 1$이면 안정성 보장
- 각 스테이지를 독립적으로 테스트 가능
- 음성 코딩(LPC)에 사용

```python
def lattice_filter_demo():
    """Demonstrate lattice FIR filter structure."""
    def fir_to_lattice(b):
        """Convert FIR coefficients to lattice (reflection) coefficients."""
        M = len(b) - 1
        a = np.array(b, dtype=float) / b[0]
        k = np.zeros(M)

        for m in range(M, 0, -1):
            k[m - 1] = a[m]
            if abs(k[m - 1]) >= 1:
                raise ValueError(f"Unstable: |k[{m-1}]| = {abs(k[m-1]):.4f} >= 1")
            # Levinson-Durbin step-down
            a_new = np.zeros(m)
            for i in range(1, m):
                a_new[i] = (a[i] - k[m - 1] * a[m - i]) / (1 - k[m - 1] ** 2)
            a = a_new

        return k

    def lattice_filter(k, x):
        """Apply lattice FIR filter with reflection coefficients k."""
        M = len(k)
        N = len(x)
        y = np.zeros(N)

        # State: g[m] for each stage
        g = np.zeros(M)

        for n in range(N):
            f = x[n]
            g_new = np.zeros(M)

            for m in range(M):
                g_old = g[m] if m < M else 0
                f_new = f + k[m] * g[m]
                g_new[m] = k[m] * f + g[m]
                f = f_new

            # Shift g states
            g[1:] = g_new[:-1]
            g[0] = x[n]

            y[n] = f

        return y

    # Example: simple FIR filter
    b = np.array([1.0, 0.5, -0.3, 0.1])

    try:
        k = fir_to_lattice(b)
        print("Lattice Filter Coefficients")
        print("=" * 40)
        print(f"FIR coefficients: {b}")
        print(f"Lattice (reflection) coefficients: {k}")
        print(f"All |k| < 1: {all(np.abs(k) < 1)}")
    except ValueError as e:
        print(f"Error: {e}")

lattice_filter_demo()
```

### 8.7 구조 비교 요약

| 구조 | 메모리 | 샘플당 곱셈 | 민감도 | 비고 |
|------|--------|------------|--------|------|
| 직접형 I | $M + N$ | $M + N + 1$ | 보통 | 단순, 직접적 |
| 직접형 II | $\max(M,N)$ | $M + N + 1$ | IIR에서 더 높음 | 정준형 (최소 메모리) |
| 전치 직접형 II | $\max(M,N)$ | $M + N + 1$ | 부동소수점에서 좋음 | 부동소수점에 선호 |
| 종속형 (SOS) | $2L$ | $5L$ | 낮음 | 고정소수점 IIR에 최적 |
| 병렬형 | $2L$ | $3L + 1$ | 낮음 | 병렬 하드웨어에 적합 |
| 격자형 | $M$ | $2M$ | 매우 낮음 | $|k|<1$로 안정성 보장 |

---

## 9. 양자화 효과(Quantization Effects)

### 9.1 양자화 오차의 원인

고정소수점 구현에서 세 가지 유형의 양자화 오차가 발생합니다:

1. **계수 양자화(Coefficient quantization)**: 필터 계수가 사용 가능한 워드 길이로 반올림됨
2. **입력 양자화(Input quantization)**: ADC 양자화 잡음
3. **산술 양자화(Arithmetic quantization)**: 곱셈 후 반올림 (곱 반올림 잡음)

### 9.2 계수 양자화

필터 계수가 양자화되면 (예: 16비트 고정소수점), 극점과 영점이 설계된 위치에서 이동합니다. 이로 인해:
- 주파수 응답이 변경됨
- 안정한 필터가 불안정해질 수 있음
- 통과대역 리플이 증가하거나 저지대역 감쇠가 감소할 수 있음

**IIR 필터는 FIR 필터보다 계수 양자화에 훨씬 민감합니다.** 분모 계수의 작은 변화가 극점의 큰 이동을 유발하며, 특히 고차 필터에서 심합니다.

```python
def coefficient_quantization_effects():
    """Demonstrate the effect of coefficient quantization on filter response."""
    # Design an 8th-order bandpass Butterworth
    order = 8
    fs = 8000
    f_low, f_high = 800, 1200  # Hz
    b, a = signal.butter(order, [f_low, f_high], btype='band', fs=fs)

    # Quantize coefficients to different bit widths
    bit_widths = [32, 16, 12, 10, 8]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    w_ref, H_ref = signal.freqz(b, a, worN=4096, fs=fs)
    axes[0].plot(w_ref, 20 * np.log10(np.abs(H_ref) + 1e-12),
                 'k-', linewidth=2, label='Float64 (reference)')

    for bits in bit_widths:
        # Quantize coefficients
        scale = 2 ** (bits - 1)
        b_q = np.round(b * scale) / scale
        a_q = np.round(a * scale) / scale

        # Check stability
        poles_q = np.roots(a_q)
        stable = all(np.abs(poles_q) < 1)

        w_q, H_q = signal.freqz(b_q, a_q, worN=4096, fs=fs)

        label = f'{bits}-bit'
        if not stable:
            label += ' (UNSTABLE!)'

        axes[0].plot(w_q, 20 * np.log10(np.abs(H_q) + 1e-12),
                     linewidth=1, label=label, alpha=0.8)

    axes[0].set_title(f'Coefficient Quantization Effect ({order}th-order Bandpass)')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].set_ylim(-80, 10)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Compare with SOS implementation (much more robust)
    sos = signal.butter(order, [f_low, f_high], btype='band', fs=fs,
                        output='sos')

    for bits in [16, 10, 8]:
        scale = 2 ** (bits - 1)
        sos_q = np.round(sos * scale) / scale

        # Check stability of each section
        stable = True
        for section in sos_q:
            poles = np.roots(section[3:])
            if any(np.abs(poles) >= 1):
                stable = False
                break

        w_q, H_q = signal.sosfreqz(sos_q, worN=4096, fs=fs)
        label = f'SOS {bits}-bit'
        if not stable:
            label += ' (UNSTABLE!)'
        axes[1].plot(w_q, 20 * np.log10(np.abs(H_q) + 1e-12),
                     linewidth=1.5, label=label)

    axes[1].plot(w_ref, 20 * np.log10(np.abs(H_ref) + 1e-12),
                 'k-', linewidth=2, label='Reference')
    axes[1].set_title('SOS (Cascade) Form: Much More Robust to Quantization')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude (dB)')
    axes[1].set_ylim(-80, 10)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('quantization_effects.png', dpi=150)
    plt.show()

coefficient_quantization_effects()
```

### 9.3 오버플로우와 한계 사이클(Limit Cycles)

**오버플로우(Overflow):** 중간 결과가 고정소수점 범위를 초과하여 wrap-around(2의 보수) 또는 포화(saturation)가 발생하는 것. 큰 오류를 방지하기 위해 포화 산술이 선호됩니다.

**한계 사이클(Limit cycles):** 고정소수점 IIR 필터에서 중간 값의 양자화로 인해 입력이 0이 된 후에도 출력이 지속적으로 진동하는 현상. 두 가지 유형이 있습니다:

1. **과립 한계 사이클(Granular limit cycles, 데드밴드 효과):** 재귀 연산의 양자화로 인한 0 근방의 소진폭 진동
2. **오버플로우 한계 사이클(Overflow limit cycles):** 피드백 경로의 산술 오버플로우로 인한 대진폭 진동

```python
def demonstrate_limit_cycles():
    """Demonstrate limit cycles in quantized IIR filters."""
    # Second-order IIR system
    # y[n] = 1.5*y[n-1] - 0.85*y[n-2] + x[n]
    a1, a2 = -1.5, 0.85

    N = 200
    bits = 8
    scale = 2 ** (bits - 1)

    # Input: impulse followed by zeros
    x = np.zeros(N)
    x[0] = 1.0

    # Float implementation (no quantization)
    y_float = np.zeros(N)
    for n in range(N):
        y_float[n] = x[n]
        if n >= 1:
            y_float[n] -= a1 * y_float[n - 1]
        if n >= 2:
            y_float[n] -= a2 * y_float[n - 2]

    # Fixed-point with rounding
    y_fixed = np.zeros(N)
    for n in range(N):
        y_val = x[n]
        if n >= 1:
            y_val -= a1 * y_fixed[n - 1]
        if n >= 2:
            y_val -= a2 * y_fixed[n - 2]
        # Quantize output
        y_fixed[n] = np.round(y_val * scale) / scale

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(np.arange(N), y_float, 'b-', linewidth=1,
                 label='Float64 (decays to zero)')
    axes[0].set_title('Floating-Point Implementation')
    axes[0].set_xlabel('n')
    axes[0].set_ylabel('y[n]')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(np.arange(N), y_fixed, 'r-', linewidth=1,
                 label=f'{bits}-bit fixed-point (may show limit cycle)')
    axes[1].set_title(f'Fixed-Point Implementation ({bits}-bit)')
    axes[1].set_xlabel('n')
    axes[1].set_ylabel('y[n]')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('limit_cycles.png', dpi=150)
    plt.show()

    # Check for limit cycle
    tail = y_fixed[-50:]
    if np.max(np.abs(tail)) > 0.5 / scale:
        print(f"Limit cycle detected! Tail amplitude: {np.max(np.abs(tail)):.6f}")
    else:
        print("No significant limit cycle in this example.")

demonstrate_limit_cycles()
```

### 9.4 완화 전략

| 문제 | 해결책 |
|------|--------|
| 계수 양자화 | 종속형 (SOS) 사용; 더 높은 정밀도 사용 |
| 오버플로우 | 포화 산술; 적절한 스케일링 |
| 과립 한계 사이클 | 디더링(Dithering); 크기 절삭(magnitude truncation) |
| 오버플로우 한계 사이클 | 포화 산술; 적절한 순서를 가진 SOS |
| 일반적 민감도 | IIR에 SOS 사용; FIR은 본질적으로 더 강인 |

---

## 10. 실용적 필터 응용 예제

```python
def practical_filtering_example():
    """Complete practical example: filtering a noisy ECG-like signal."""
    fs = 500  # Hz (typical ECG sampling rate)
    T = 5.0   # 5 seconds
    N = int(fs * T)
    t = np.arange(N) / fs

    # Simulate ECG-like signal (simplified)
    ecg = np.zeros(N)
    heart_rate = 72  # BPM
    beat_period = fs * 60 / heart_rate

    for beat in range(int(T * heart_rate / 60) + 1):
        beat_start = int(beat * beat_period)
        if beat_start + 50 < N:
            # QRS complex (simplified as a narrow Gaussian)
            idx = np.arange(max(0, beat_start - 25), min(N, beat_start + 25))
            ecg[idx] += np.exp(-0.5 * ((idx - beat_start) / 3) ** 2)
            # T-wave
            if beat_start + 80 < N:
                idx_t = np.arange(beat_start + 40, min(N, beat_start + 80))
                ecg[idx_t] += 0.3 * np.exp(-0.5 * ((idx_t - beat_start - 60) / 10) ** 2)

    # Add noise
    powerline = 0.3 * np.sin(2 * np.pi * 50 * t)  # 50 Hz powerline
    high_freq = 0.1 * np.random.randn(N)  # High-frequency noise
    baseline = 0.2 * np.sin(2 * np.pi * 0.3 * t)  # Baseline wander
    noisy_ecg = ecg + powerline + high_freq + baseline

    # Design filters

    # 1. Notch filter to remove 50 Hz powerline (IIR)
    f_notch = 50
    Q = 30  # Quality factor
    b_notch, a_notch = signal.iirnotch(f_notch, Q, fs=fs)

    # 2. Bandpass filter for ECG (0.5-40 Hz) (FIR)
    b_bp = signal.firwin(101, [0.5, 40], pass_zero=False, fs=fs)

    # 3. Alternative: Butterworth bandpass (IIR)
    b_bp_iir, a_bp_iir = signal.butter(4, [0.5, 40], btype='band', fs=fs)

    # Apply filters
    y_notch = signal.filtfilt(b_notch, a_notch, noisy_ecg)
    y_fir_bp = signal.filtfilt(b_bp, [1.0], y_notch)
    y_iir_bp = signal.filtfilt(b_bp_iir, a_bp_iir, noisy_ecg)

    # Plot results
    fig, axes = plt.subplots(4, 1, figsize=(16, 14))

    time_range = (1.0, 3.0)  # Show 2 seconds
    mask = (t >= time_range[0]) & (t <= time_range[1])

    axes[0].plot(t[mask], ecg[mask], 'b-', linewidth=1)
    axes[0].set_title('Clean ECG Signal')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t[mask], noisy_ecg[mask], 'r-', linewidth=0.5)
    axes[1].set_title('Noisy ECG (50 Hz + HF noise + baseline wander)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t[mask], y_fir_bp[mask], 'g-', linewidth=1, label='FIR bandpass')
    axes[2].plot(t[mask], ecg[mask], 'b--', linewidth=0.5, alpha=0.5,
                 label='Original')
    axes[2].set_title('Filtered (Notch + FIR Bandpass)')
    axes[2].set_ylabel('Amplitude')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t[mask], y_iir_bp[mask], 'm-', linewidth=1,
                 label='IIR bandpass')
    axes[3].plot(t[mask], ecg[mask], 'b--', linewidth=0.5, alpha=0.5,
                 label='Original')
    axes[3].set_title('Filtered (IIR Butterworth Bandpass)')
    axes[3].set_ylabel('Amplitude')
    axes[3].set_xlabel('Time (s)')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ecg_filtering.png', dpi=150)
    plt.show()

    # Compare filter responses
    fig, ax = plt.subplots(figsize=(12, 5))
    w_n, H_n = signal.freqz(b_notch, a_notch, worN=4096, fs=fs)
    w_f, H_f = signal.freqz(b_bp, [1.0], worN=4096, fs=fs)
    w_i, H_i = signal.freqz(b_bp_iir, a_bp_iir, worN=4096, fs=fs)

    ax.plot(w_n, 20 * np.log10(np.abs(H_n) + 1e-12), 'r-',
            label='50 Hz Notch')
    ax.plot(w_f, 20 * np.log10(np.abs(H_f) + 1e-12), 'g-',
            label='FIR Bandpass (0.5-40 Hz)')
    ax.plot(w_i, 20 * np.log10(np.abs(H_i) + 1e-12), 'm-',
            label='IIR Bandpass (0.5-40 Hz)')
    ax.set_title('Filter Frequency Responses')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_xlim(0, 100)
    ax.set_ylim(-60, 5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ecg_filter_responses.png', dpi=150)
    plt.show()

practical_filtering_example()
```

---

## 11. 요약

```
┌─────────────────────────────────────────────────────────────────┐
│               Digital Filter Fundamentals                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Filter Types:                                                   │
│    FIR: y[n] = Σ bk x[n-k]   (non-recursive, always stable)   │
│    IIR: y[n] = Σ bk x[n-k] - Σ ak y[n-k]   (recursive)       │
│                                                                  │
│  Transfer Function:                                              │
│    H(z) = B(z)/A(z)                                             │
│    FIR: H(z) = polynomial (all-zero)                            │
│    IIR: H(z) = rational (poles and zeros)                       │
│                                                                  │
│  Linear Phase:                                                   │
│    FIR with symmetric/anti-symmetric coefficients               │
│    4 types (I-IV) with different constraints                    │
│    Constant group delay → no waveform distortion                │
│                                                                  │
│  Specifications:                                                 │
│    Passband ripple δp, Stopband attenuation δs                  │
│    Transition width Δω = ωs - ωp                                │
│    Trade-off: sharper transition → higher order                 │
│                                                                  │
│  Filter Structures:                                              │
│    Direct Form I/II → simple but sensitive                      │
│    Transposed DF II → better for floating-point                 │
│    Cascade (SOS) → robust to quantization (preferred!)          │
│    Lattice → inherent stability check via |k| < 1              │
│                                                                  │
│  Quantization Effects:                                           │
│    Coefficient quantization → pole/zero shift                   │
│    Arithmetic rounding → noise floor                            │
│    Overflow → saturation arithmetic needed                      │
│    Limit cycles → IIR-specific problem                          │
│    Mitigation: use SOS form + adequate word length              │
│                                                                  │
│  FIR vs. IIR Decision Guide:                                    │
│    Need linear phase? → FIR                                     │
│    Need minimum order? → IIR                                    │
│    Fixed-point implementation? → FIR or SOS-IIR                 │
│    Guaranteed stability? → FIR                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 12. 연습 문제

### 연습 1: FIR 필터 분석

FIR 필터 $h[n] = \{1, -2, 3, -2, 1\}$이 주어졌을 때:

**(a)** 차분 방정식과 전달 함수 $H(z)$를 작성하시오.

**(b)** 모든 영점을 구하고 그래프로 나타내시오. 특정 대칭 패턴이 있는지 검증하시오.

**(c)** 크기 및 위상 응답을 그래프로 나타내시오. 이 필터는 선형 위상을 가지는가?

**(d)** 필터 유형 (I, II, III, IV)을 분류하고, 수행할 수 있는 주파수 선택 필터링 유형 (저역통과, 고역통과, 대역통과, 대역저지)을 식별하시오.

**(e)** 신호 $x[n] = \cos(0.2\pi n) + \cos(0.8\pi n)$을 필터링하고 결과를 설명하시오.

### 연습 2: IIR 필터 설계 및 분석

8 kHz 샘플링, 0.5 dB 통과대역 리플, 2 kHz 차단 주파수로 4차 Chebyshev Type I 저역통과 필터를 설계하시오.

**(a)** `scipy.signal.cheby1`을 사용하여 필터 계수를 구하시오.

**(b)** 극점-영점도를 그리고 모든 극점이 단위원 내에 있는지 검증하시오.

**(c)** 크기 응답, 위상 응답, 군지연을 그래프로 나타내시오.

**(d)** 동일한 차수와 차단 주파수를 가진 Butterworth 필터와 비교하시오. 어느 필터가 더 좋은 저지대역 감쇠를 가지는가? 어느 필터의 군지연이 더 일정한가?

**(e)** SOS 형식으로 변환하고 주파수 응답이 일치하는지 검증하시오.

### 연습 3: 선형 위상 검증

**(a)** fs = 8000 Hz에서 통과대역 1000-2000 Hz인 Type I 선형 위상 FIR 대역통과 필터 (51탭)를 설계하시오.

**(b)** Type III 선형 위상 FIR 미분기 (51탭)를 설계하시오.

**(c)** 각 필터에 대해 수치적으로 다음을 검증하시오:
- 계수가 예상되는 대칭 조건을 만족하는지
- 군지연이 모든 주파수에서 일정한지
- 위상 응답이 선형 (또는 $\pi$만큼 점프를 가진 구간별 선형)인지

### 연습 4: 필터 구조 구현

`scipy.signal.lfilter`를 사용하지 않고 다음을 처음부터 구현하시오:

**(a)** 필터 $H(z) = \frac{1 + 0.5z^{-1}}{1 - 0.9z^{-1} + 0.81z^{-2}}$에 대한 직접형 I

**(b)** 동일한 필터에 대한 직접형 II

**(c)** 두 개의 바이쿼드 섹션을 사용한 종속형

**(d)** 1000샘플 테스트 신호를 세 가지 구현으로 처리하고, 결과가 (부동소수점 정밀도 내에서) 동일한지 검증하시오.

### 연습 5: 양자화 실험

8000 Hz 샘플링에서 300-3400 Hz의 10차 IIR 타원 대역통과 필터에 대해:

**(a)** 16비트 계수 양자화로 직접형 II를 구현하시오. 안정한가?

**(b)** 16비트 양자화로 동일한 필터를 종속형 (SOS)으로 구현하시오. 안정한가?

**(c)** 두 양자화 구현의 주파수 응답을 기준 (float64)과 비교하여 그래프로 나타내시오. 어느 구조가 원하는 응답을 더 잘 보존하는가?

**(d)** 각 구조에서 안정성을 위한 최소 비트 수를 구하시오.

### 연습 6: 실시간 필터 시뮬레이션

샘플 단위 필터링 함수(실시간 처리 시뮬레이션)를 구현하시오:

```python
class RealtimeFilter:
    def __init__(self, b, a):
        # Initialize state
        pass

    def process_sample(self, x_n):
        # Process one sample, return one output sample
        pass
```

**(a)** 전치 직접형 II를 사용하여 구현하시오.

**(b)** 스트리밍 입력으로 테스트하고 (샘플을 하나씩 처리) `scipy.signal.lfilter`의 일괄 처리와 결과가 일치하는지 검증하시오.

**(c)** 다양한 필터 차수 (4, 8, 16, 32)에서 샘플당 실행 시간을 측정하시오.

### 연습 7: 다중 레이트 필터 뱅크

fs = 8 kHz에서 신호를 저역 (0-1 kHz), 중역 (1-3 kHz), 고역 (3-4 kHz) 대역으로 분할하는 3대역 필터 뱅크를 설계하시오:

**(a)** 적절한 대역통과 필터를 설계하시오 (FIR 또는 IIR을 선택하고 이유를 설명하시오).

**(b)** 세 대역 모두에 성분을 포함하는 테스트 신호에 필터 뱅크를 적용하시오.

**(c)** 세 개의 필터링된 출력을 합산하여 원본 신호를 재구성하시오.

**(d)** 재구성 오차를 측정하시오. 완벽한 재구성이 이루어지지 않는 원인은 무엇인가?

---

## 13. 추가 참고 자료

- Oppenheim, Schafer. *Discrete-Time Signal Processing*, 3rd ed. Chapters 5-6.
- Proakis, Manolakis. *Digital Signal Processing*, 4th ed. Chapters 7-9.
- Mitra, S. K. *Digital Signal Processing: A Computer-Based Approach*, 4th ed. Chapters 8-9.
- Smith, S. W. *The Scientist and Engineer's Guide to DSP*, Chapters 14-20.
- Lyons, R. G. *Understanding Digital Signal Processing*, 3rd ed. Chapters 5-7.
- Jackson, L. B. *Digital Filters and Signal Processing*, Chapter 11 (양자화 효과).

---

**이전**: [07. Z-변환](07_Z_Transform.md) | **다음**: [09. FIR 필터 설계](09_FIR_Filter_Design.md)
