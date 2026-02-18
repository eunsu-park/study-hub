# IIR 필터 설계

## 학습 목표

- 고전적인 아날로그 원형 필터(버터워스(Butterworth), 체비쇼프(Chebyshev), 타원형(Elliptic))의 특성 이해
- 아날로그-디지털 변환을 위한 쌍일차 변환(bilinear transform)과 임펄스 불변 방법(impulse invariance) 마스터
- 주파수 영역 규격으로부터 필터 차수 결정 방법 학습
- Python의 `scipy.signal` 모듈을 사용한 IIR 디지털 필터 설계
- 극-영점 분석(pole-zero analysis)을 통한 필터 안정성 검증
- 필터 유형 비교 및 특정 응용에 적합한 설계 선택

---

## 목차

1. [IIR 필터 설계 소개](#1-iir-필터-설계-소개)
2. [아날로그 원형 필터](#2-아날로그-원형-필터)
3. [버터워스 필터](#3-버터워스-필터)
4. [체비쇼프 Type I 필터](#4-체비쇼프-type-i-필터)
5. [체비쇼프 Type II 필터](#5-체비쇼프-type-ii-필터)
6. [타원형(코어) 필터](#6-타원형코어-필터)
7. [아날로그-디지털 변환](#7-아날로그-디지털-변환)
8. [쌍일차 변환](#8-쌍일차-변환)
9. [임펄스 불변 방법](#9-임펄스-불변-방법)
10. [완전한 IIR 설계 절차](#10-완전한-iir-설계-절차)
11. [안정성 분석](#11-안정성-분석)
12. [필터 유형 비교](#12-필터-유형-비교)
13. [Python 구현](#13-python-구현)
14. [연습 문제](#14-연습-문제)

---

## 1. IIR 필터 설계 소개

### 1.1 IIR 필터 구조

IIR(무한 임펄스 응답(Infinite Impulse Response)) 필터는 순방향(feedforward)과 피드백(feedback) 경로를 모두 갖습니다. 전달 함수는 다항식의 비율로 나타냅니다:

$$H(z) = \frac{B(z)}{A(z)} = \frac{\sum_{k=0}^{M} b_k z^{-k}}{1 + \sum_{k=1}^{N} a_k z^{-k}}$$

차분 방정식(difference equation)은 다음과 같습니다:

$$y[n] = \sum_{k=0}^{M} b_k x[n-k] - \sum_{k=1}^{N} a_k y[n-k]$$

### 1.2 설계 접근법

FIR 필터(이산 영역에서 직접 설계)와 달리, IIR 필터는 일반적으로 다음과 같이 설계합니다:

1. **잘 알려진 특성을 갖는 아날로그 원형** $H_a(s)$에서 시작
2. 아날로그 필터를 디지털 필터 $H(z)$로 **변환**

```
┌──────────────────────────────────────────────────────────────────┐
│                 IIR Filter Design Pipeline                       │
│                                                                  │
│  Digital Specs     Analog Specs     Analog Filter    Digital     │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐    ┌────────┐  │
│  │ ωp, ωs   │ ──▶ │ Ωp, Ωs   │ ──▶ │ Ha(s)    │ ──▶│ H(z)   │  │
│  │ δ₁, δ₂   │     │ δ₁, δ₂   │     │          │    │        │  │
│  └──────────┘     └──────────┘     └──────────┘    └────────┘  │
│                                                                  │
│  Step 1:           Step 2:          Step 3:         Step 4:     │
│  Specify digital   Pre-warp to      Design analog   Apply       │
│  requirements      analog specs     prototype       BLT/IIM    │
└──────────────────────────────────────────────────────────────────┘
```

### 1.3 아날로그 원형에서 시작하는 이유

- **수십 년의 이론**: 버터워스(Butterworth), 체비쇼프(Chebyshev), 타원형 필터 설계는 잘 정립되어 있음
- **닫힌 형태의 해**: 극점(pole)/영점(zero) 위치를 해석적으로 계산 가능
- **최적 특성**: 각 유형은 특정 의미에서 최적 (평탄도, 등리플, 최소 차수)
- **주파수 변환**: 저역 통과(lowpass) 원형을 고역 통과(highpass), 대역 통과(bandpass), 대역 저지(bandstop)로 변환 가능

---

## 2. 아날로그 원형 필터

### 2.1 아날로그 필터 규격

아날로그 저역 통과 필터 규격은 다음으로 구성됩니다:

- **통과대역 엣지 주파수(passband edge frequency)** $\Omega_p$
- **저지대역 엣지 주파수(stopband edge frequency)** $\Omega_s$
- **통과대역 리플(passband ripple)** $R_p$ (dB) 또는 허용오차 $\epsilon$
- **저지대역 감쇠(stopband attenuation)** $A_s$ (dB)

리플 파라미터 $\epsilon$과 통과대역 리플의 관계:

$$R_p = 10\log_{10}(1 + \epsilon^2) \quad \text{dB}$$

### 2.2 정규화된 저역 통과 원형

모든 고전적 설계는 통과대역 엣지가 $\Omega_p = 1$ rad/s인 **정규화된 저역 통과 원형(normalized lowpass prototype)**에서 시작합니다. 이후 원형을 주파수 스케일링하고 원하는 필터 유형으로 변환합니다.

**선택도 인수(selectivity factor):**

$$k = \frac{\Omega_p}{\Omega_s} < 1$$

$k$가 작을수록 전이대역(transition band)이 통과대역에 비해 넓어져 설계가 쉬워집니다.

---

## 3. 버터워스 필터

### 3.1 크기 응답

버터워스(Butterworth) 필터는 통과대역에서 **최대 평탄 크기 응답(maximally flat magnitude response)**을 제공합니다. 제곱 크기 응답은:

$$|H_a(j\Omega)|^2 = \frac{1}{1 + \left(\Omega / \Omega_c\right)^{2N}}$$

여기서 $N$은 필터 차수, $\Omega_c$는 $-3$ dB 차단 주파수입니다.

**특성:**
- $|H_a(j\Omega)|^2$의 $2N-1$차까지 모든 미분값이 $\Omega = 0$에서 0
- 단조 감소(monotonically decreasing) 크기
- 차단 주파수에서 $|H_a(j\Omega_c)|^2 = 1/2$ ($-3$ dB)
- 롤오프(rolloff) 속도: 저지대역에서 $-20N$ dB/decade

### 3.2 극점 위치

버터워스 필터의 극점은 $s$-평면에서 반지름 $\Omega_c$인 원 위에 등간격으로 위치합니다:

$$s_k = \Omega_c \exp\left[j\frac{\pi}{2N}(2k + N - 1)\right], \quad k = 0, 1, \ldots, 2N-1$$

안정적(인과적) 필터를 위해 좌반평면(left-half-plane) 극점($\text{Re}(s_k) < 0$)만 선택합니다.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def butterworth_poles(N, Omega_c=1.0):
    """Compute Butterworth filter poles in the s-plane."""
    poles = []
    for k in range(2 * N):
        s_k = Omega_c * np.exp(1j * np.pi * (2 * k + N - 1) / (2 * N))
        if np.real(s_k) < 0:  # Left-half plane only
            poles.append(s_k)
    return np.array(poles)

# Visualize pole locations for different orders
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, N in zip(axes, [2, 4, 8]):
    poles = butterworth_poles(N)
    theta = np.linspace(0, 2 * np.pi, 200)

    # Unit circle
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
    # Poles
    ax.plot(np.real(poles), np.imag(poles), 'rx', markersize=12, markeredgewidth=2)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title(f'Butterworth N={N} Poles')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.5, 0.5)
    ax.set_ylim(-1.5, 1.5)

plt.tight_layout()
plt.savefig('butterworth_poles.png', dpi=150)
plt.close()
```

### 3.3 차수 결정

규격 $(\Omega_p, \Omega_s, R_p, A_s)$가 주어지면 최소 버터워스 차수는:

$$N \geq \frac{\log\left(\frac{10^{A_s/10} - 1}{10^{R_p/10} - 1}\right)}{2\log(\Omega_s / \Omega_p)}$$

```python
def butterworth_order(Omega_p, Omega_s, Rp_dB, As_dB):
    """Compute minimum Butterworth filter order."""
    numerator = np.log10((10**(As_dB / 10) - 1) / (10**(Rp_dB / 10) - 1))
    denominator = 2 * np.log10(Omega_s / Omega_p)
    N = int(np.ceil(numerator / denominator))
    return N

# Example
N = butterworth_order(1.0, 2.0, 1.0, 40)
print(f"Minimum Butterworth order: N = {N}")
# Compare with scipy
N_scipy, Wn = signal.buttord(1.0, 2.0, 1.0, 40, analog=True)
print(f"scipy.signal.buttord: N = {N_scipy}, Wn = {Wn:.4f}")
```

### 3.4 크기 응답 시각화

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Different orders
orders = [1, 2, 4, 8, 16]
for N in orders:
    b, a = signal.butter(N, 1.0, analog=True)
    w, H = signal.freqs(b, a, worN=np.logspace(-1, 1, 1000))
    H_dB = 20 * np.log10(np.abs(H))

    axes[0].plot(w, H_dB, linewidth=1.5, label=f'N={N}')
    axes[1].plot(w, np.abs(H), linewidth=1.5, label=f'N={N}')

axes[0].set_xlabel('Frequency (rad/s)')
axes[0].set_ylabel('Magnitude (dB)')
axes[0].set_title('Butterworth: Magnitude (dB)')
axes[0].set_xlim(0.1, 10)
axes[0].set_ylim(-80, 5)
axes[0].axhline(-3, color='gray', linestyle=':', alpha=0.5, label='-3 dB')
axes[0].axvline(1.0, color='gray', linestyle=':', alpha=0.5)
axes[0].set_xscale('log')
axes[0].legend()
axes[0].grid(True, which='both', alpha=0.3)

axes[1].set_xlabel('Frequency (rad/s)')
axes[1].set_ylabel('|H(jΩ)|')
axes[1].set_title('Butterworth: Magnitude (Linear)')
axes[1].set_xlim(0, 3)
axes[1].axhline(1/np.sqrt(2), color='gray', linestyle=':', alpha=0.5, label='-3 dB')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('butterworth_responses.png', dpi=150)
plt.close()
```

---

## 4. 체비쇼프 Type I 필터

### 4.1 크기 응답

체비쇼프(Chebyshev) Type I 필터는 **등리플 통과대역(equiripple passband)**과 단조 저지대역을 갖습니다:

$$|H_a(j\Omega)|^2 = \frac{1}{1 + \epsilon^2 T_N^2(\Omega / \Omega_p)}$$

여기서 $T_N(\cdot)$는 $N$차 제1종 체비쇼프 다항식(Chebyshev polynomial of the first kind)입니다:

$$T_N(x) = \begin{cases} \cos(N \cos^{-1}(x)), & |x| \leq 1 \\ \cosh(N \cosh^{-1}(x)), & |x| > 1 \end{cases}$$

**특성:**
- 리플 $\epsilon$의 등리플 통과대역
- 통과대역 리플: $R_p = 10\log_{10}(1 + \epsilon^2)$ dB
- 동일 차수에서 버터워스보다 급격한 전이
- 단조 감소 저지대역

### 4.2 체비쇼프 다항식

처음 몇 개의 체비쇼프 다항식:

$$T_0(x) = 1, \quad T_1(x) = x, \quad T_2(x) = 2x^2 - 1$$

$$T_3(x) = 4x^3 - 3x, \quad T_4(x) = 8x^4 - 8x^2 + 1$$

점화식(recurrence relation): $T_{N+1}(x) = 2x \cdot T_N(x) - T_{N-1}(x)$

### 4.3 차수 결정

$$N \geq \frac{\cosh^{-1}\left(\sqrt{\frac{10^{A_s/10} - 1}{10^{R_p/10} - 1}}\right)}{\cosh^{-1}(\Omega_s / \Omega_p)}$$

### 4.4 극점 위치

체비쇼프 Type I 필터의 극점은 $s$-평면에서 타원 위에 위치합니다:

$$s_k = \sigma_k + j\omega_k$$

여기서:

$$\sigma_k = -\sinh\left(\frac{1}{N}\sinh^{-1}\left(\frac{1}{\epsilon}\right)\right) \sin\left(\frac{(2k-1)\pi}{2N}\right)$$

$$\omega_k = \cosh\left(\frac{1}{N}\sinh^{-1}\left(\frac{1}{\epsilon}\right)\right) \cos\left(\frac{(2k-1)\pi}{2N}\right)$$

$k = 1, 2, \ldots, N$에 대해.

```python
def chebyshev1_analysis(N, Rp_dB=1.0):
    """Analyze Chebyshev Type I filter."""
    epsilon = np.sqrt(10**(Rp_dB / 10) - 1)

    # Pole locations
    poles = []
    for k in range(1, N + 1):
        theta_k = (2 * k - 1) * np.pi / (2 * N)
        sigma_k = -np.sinh(np.arcsinh(1 / epsilon) / N) * np.sin(theta_k)
        omega_k = np.cosh(np.arcsinh(1 / epsilon) / N) * np.cos(theta_k)
        poles.append(sigma_k + 1j * omega_k)

    return np.array(poles), epsilon

# Compare Butterworth and Chebyshev
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for N in [2, 4, 6, 8]:
    # Chebyshev Type I (1 dB ripple)
    b, a = signal.cheby1(N, 1.0, 1.0, analog=True)
    w, H = signal.freqs(b, a, worN=np.logspace(-1, 1, 1000))
    axes[0].plot(w, 20 * np.log10(np.abs(H)), linewidth=1.5, label=f'N={N}')

axes[0].set_xlabel('Frequency (rad/s)')
axes[0].set_ylabel('Magnitude (dB)')
axes[0].set_title('Chebyshev Type I (Rp = 1 dB)')
axes[0].set_xlim(0.1, 10)
axes[0].set_ylim(-80, 5)
axes[0].set_xscale('log')
axes[0].legend()
axes[0].grid(True, which='both', alpha=0.3)

# Pole-zero plot for N=4
poles_butter = butterworth_poles(4)
poles_cheby, _ = chebyshev1_analysis(4, 1.0)

theta = np.linspace(0, 2 * np.pi, 200)
axes[1].plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label='Unit circle')
axes[1].plot(np.real(poles_butter), np.imag(poles_butter), 'bx',
             markersize=12, markeredgewidth=2, label='Butterworth')
axes[1].plot(np.real(poles_cheby), np.imag(poles_cheby), 'r+',
             markersize=12, markeredgewidth=2, label='Chebyshev I')
axes[1].set_xlabel('Real')
axes[1].set_ylabel('Imaginary')
axes[1].set_title('Pole Locations: Butterworth vs Chebyshev I (N=4)')
axes[1].set_aspect('equal')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(-1.5, 0.5)
axes[1].set_ylim(-1.5, 1.5)

plt.tight_layout()
plt.savefig('chebyshev1_analysis.png', dpi=150)
plt.close()
```

---

## 5. 체비쇼프 Type II 필터

### 5.1 크기 응답

체비쇼프 Type II(역 체비쇼프(inverse Chebyshev)) 필터는 **평탄한 통과대역**과 **등리플 저지대역**을 갖습니다:

$$|H_a(j\Omega)|^2 = \frac{1}{1 + \frac{1}{\epsilon^2 T_N^2(\Omega_s / \Omega)}}$$

**특성:**
- 단조 감소(평탄) 통과대역
- 저지대역에서 등리플 동작
- 극점과 영점 모두 존재 (Type I은 극점만)
- 영점이 $j\Omega$ 축에 위치하여 저지대역에 노치(null) 생성

### 5.2 Type I과의 비교

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

N = 5

# Chebyshev Type I (1 dB ripple)
b1, a1 = signal.cheby1(N, 1.0, 1.0, analog=True)
w1, H1 = signal.freqs(b1, a1, worN=np.logspace(-1, 1, 2000))

# Chebyshev Type II (40 dB stopband attenuation)
b2, a2 = signal.cheby2(N, 40, 1.0, analog=True)
w2, H2 = signal.freqs(b2, a2, worN=np.logspace(-1, 1, 2000))

# Butterworth (same order)
b_bw, a_bw = signal.butter(N, 1.0, analog=True)
w_bw, H_bw = signal.freqs(b_bw, a_bw, worN=np.logspace(-1, 1, 2000))

# dB plot
for w, H, name, style in [(w_bw, H_bw, 'Butterworth', 'b-'),
                            (w1, H1, 'Chebyshev I', 'r-'),
                            (w2, H2, 'Chebyshev II', 'g-')]:
    axes[0].plot(w, 20 * np.log10(np.abs(H) + 1e-15), style,
                 linewidth=1.5, label=name)

axes[0].set_xlabel('Frequency (rad/s)')
axes[0].set_ylabel('Magnitude (dB)')
axes[0].set_title(f'Filter Comparison (N={N})')
axes[0].set_xlim(0.1, 10)
axes[0].set_ylim(-80, 5)
axes[0].set_xscale('log')
axes[0].legend()
axes[0].grid(True, which='both', alpha=0.3)

# Passband detail (linear scale)
for w, H, name, style in [(w_bw, H_bw, 'Butterworth', 'b-'),
                            (w1, H1, 'Chebyshev I', 'r-'),
                            (w2, H2, 'Chebyshev II', 'g-')]:
    axes[1].plot(w, np.abs(H), style, linewidth=1.5, label=name)

axes[1].set_xlabel('Frequency (rad/s)')
axes[1].set_ylabel('|H(jΩ)|')
axes[1].set_title('Passband Detail')
axes[1].set_xlim(0, 2)
axes[1].set_ylim(0, 1.2)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chebyshev2_comparison.png', dpi=150)
plt.close()
```

---

## 6. 타원형(코어) 필터

### 6.1 크기 응답

타원형(Elliptic, Cauer) 필터는 통과대역과 저지대역 모두에 리플을 분산시켜 주어진 규격에 대해 **최소 차수**를 달성합니다:

$$|H_a(j\Omega)|^2 = \frac{1}{1 + \epsilon^2 R_N^2(\Omega / \Omega_p)}$$

여기서 $R_N(\cdot)$은 유리 체비쇼프(야코비 타원(Jacobian elliptic)) 함수입니다.

**특성:**
- 통과대역과 저지대역 모두 등리플
- 주어진 차수에 대해 가장 급격한 전이
- 차수 대 규격 면에서 가장 효율적
- 극점과 영점 모두 존재

### 6.2 차수 우위

동일한 규격에 대해 필요한 필터 차수는:

$$N_\text{Elliptic} \leq N_\text{Chebyshev} \leq N_\text{Butterworth}$$

```python
def compare_filter_orders(Rp_dB, As_dB, wp, ws):
    """Compare required orders for different filter types."""
    # Butterworth
    N_butter, Wn_butter = signal.buttord(wp, ws, Rp_dB, As_dB, analog=True)

    # Chebyshev Type I
    N_cheby1, Wn_cheby1 = signal.cheb1ord(wp, ws, Rp_dB, As_dB, analog=True)

    # Chebyshev Type II
    N_cheby2, Wn_cheby2 = signal.cheb2ord(wp, ws, Rp_dB, As_dB, analog=True)

    # Elliptic
    N_ellip, Wn_ellip = signal.ellipord(wp, ws, Rp_dB, As_dB, analog=True)

    print(f"Specifications: Rp={Rp_dB} dB, As={As_dB} dB, wp={wp}, ws={ws}")
    print(f"{'Filter Type':<20} {'Order N':<10} {'Natural Freq':>12}")
    print("-" * 45)
    print(f"{'Butterworth':<20} {N_butter:<10} {Wn_butter:>12.4f}")
    print(f"{'Chebyshev I':<20} {N_cheby1:<10} {Wn_cheby1:>12.4f}")
    print(f"{'Chebyshev II':<20} {N_cheby2:<10} {Wn_cheby2:>12.4f}")
    print(f"{'Elliptic':<20} {N_ellip:<10} {Wn_ellip:>12.4f}")

    return N_butter, N_cheby1, N_cheby2, N_ellip

orders = compare_filter_orders(1.0, 60, 1.0, 1.5)
```

### 6.3 타원형 필터 시각화

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Same specifications, different filter types
Rp = 1.0  # dB
As = 60   # dB
wp = 2 * np.pi * 1000  # rad/s
ws = 2 * np.pi * 1300  # rad/s

filters = [
    ('Butterworth', signal.buttord, signal.butter),
    ('Chebyshev I', signal.cheb1ord, signal.cheby1),
    ('Chebyshev II', signal.cheb2ord, signal.cheby2),
    ('Elliptic', signal.ellipord, signal.ellip),
]

for ax, (name, ord_func, design_func) in zip(axes.flat, filters):
    N, Wn = ord_func(wp, ws, Rp, As, analog=True)

    if 'Chebyshev I' in name:
        b, a = design_func(N, Rp, Wn, analog=True)
    elif 'Chebyshev II' in name:
        b, a = design_func(N, As, Wn, analog=True)
    elif 'Elliptic' in name:
        b, a = design_func(N, Rp, As, Wn, analog=True)
    else:
        b, a = design_func(N, Wn, analog=True)

    w, H = signal.freqs(b, a, worN=np.linspace(0, 3 * ws, 5000))
    H_dB = 20 * np.log10(np.abs(H) + 1e-15)

    ax.plot(w / (2 * np.pi), H_dB, 'b-', linewidth=1.5)
    ax.axhline(-Rp, color='g', linestyle='--', alpha=0.5, label=f'-{Rp} dB')
    ax.axhline(-As, color='r', linestyle='--', alpha=0.5, label=f'-{As} dB')
    ax.axvline(wp / (2 * np.pi), color='g', linestyle=':', alpha=0.5)
    ax.axvline(ws / (2 * np.pi), color='r', linestyle=':', alpha=0.5)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title(f'{name} (N={N})')
    ax.set_ylim(-80, 5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle(f'Filter Type Comparison (Rp={Rp} dB, As={As} dB)', fontsize=13)
plt.tight_layout()
plt.savefig('filter_type_comparison.png', dpi=150)
plt.close()
```

---

## 7. 아날로그-디지털 변환

### 7.1 방법 개요

아날로그 필터 $H_a(s)$를 디지털 필터 $H(z)$로 변환하는 세 가지 주요 방법:

```
┌─────────────────────────────────────────────────────────────────┐
│            Analog-to-Digital Conversion Methods                  │
├──────────────────┬──────────────────────────────────────────────┤
│ Method           │ Mapping s → z                                │
├──────────────────┼──────────────────────────────────────────────┤
│ Bilinear         │ s = (2/T)(z-1)/(z+1)                        │
│ Transform        │ - No aliasing                                │
│                  │ - Frequency warping (correctable)            │
│                  │ - Most widely used                           │
├──────────────────┼──────────────────────────────────────────────┤
│ Impulse          │ h[n] = T · h_a(nT)                           │
│ Invariance       │ - Preserves impulse response shape           │
│                  │ - Aliasing in frequency domain               │
│                  │ - Only for bandlimited filters (LP, BP)      │
├──────────────────┼──────────────────────────────────────────────┤
│ Matched          │ Map poles: s_k → z_k = e^(s_k T)             │
│ Z-Transform      │ Map zeros: same mapping                      │
│                  │ - Simple but no formal optimality            │
└──────────────────┴──────────────────────────────────────────────┘
```

---

## 8. 쌍일차 변환

### 8.1 매핑

쌍일차 변환(bilinear transform, BLT)은 $s$-평면 전체를 $z$-평면으로 다음과 같이 매핑합니다:

$$s = \frac{2}{T} \cdot \frac{z - 1}{z + 1}$$

또는 동등하게:

$$z = \frac{1 + (T/2)s}{1 - (T/2)s}$$

여기서 $T$는 샘플링 주기입니다.

### 8.2 주요 특성

**안정성 보존**: $s$-평면의 좌반평면이 $z$-평면의 단위원 내부로 매핑됩니다. 이를 통해 안정적인 아날로그 필터는 항상 안정적인 디지털 필터를 생성합니다.

**주파수 매핑**: 허수축($s = j\Omega$)은 비선형 주파수 워핑(frequency warping)으로 단위원($z = e^{j\omega}$)으로 매핑됩니다:

$$\omega = 2\arctan\left(\frac{\Omega T}{2}\right) \quad \Leftrightarrow \quad \Omega = \frac{2}{T}\tan\left(\frac{\omega}{2}\right)$$

### 8.3 주파수 워핑

```
Bilinear Transform Frequency Warping:

Ω (analog)     ω (digital)
   ↑              ↑
   │     ╱       π│─────────────────╱
   │   ╱          │              ╱
   │  ╱           │            ╱
   │ ╱            │         ╱      Nonlinear
   │╱             │      ╱        compression
 0 ├───→ ω     0 ├───╱──────────→ Ω
   0    π         0

- Low frequencies: Ω ≈ ω (nearly linear)
- High frequencies: compressed into [0, π]
- Ω = ∞ maps to ω = π
```

```python
# Visualize frequency warping
T = 1.0  # Sampling period
omega = np.linspace(0, np.pi - 0.01, 1000)
Omega = (2 / T) * np.tan(omega / 2)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(Omega, omega / np.pi, 'b-', linewidth=2, label='BLT mapping')
axes[0].plot(Omega, Omega * T / np.pi, 'r--', linewidth=1.5, label='Linear (ideal)')
axes[0].set_xlabel('Analog Frequency Ω (rad/s)')
axes[0].set_ylabel('Digital Frequency ω/π')
axes[0].set_title('Bilinear Transform: Frequency Warping')
axes[0].set_xlim(0, 20)
axes[0].set_ylim(0, 1.1)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Warping effect at different frequencies
freq_analog = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
freq_digital = 2 * np.arctan(freq_analog * T / 2)

axes[1].bar(range(len(freq_analog)),
            freq_digital / (freq_analog * T) * 100 - 100,
            tick_label=[f'{f:.1f}' for f in freq_analog])
axes[1].set_xlabel('Analog Frequency Ω (rad/s)')
axes[1].set_ylabel('Warping Error (%)')
axes[1].set_title('Frequency Warping Error')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bilinear_warping.png', dpi=150)
plt.close()
```

### 8.4 사전 워핑(Pre-warping)

임계 주파수가 올바르게 매핑되도록 하기 위해 아날로그 규격을 **사전 워핑(pre-warp)**합니다:

$$\Omega_p' = \frac{2}{T}\tan\left(\frac{\omega_p}{2}\right), \quad \Omega_s' = \frac{2}{T}\tan\left(\frac{\omega_s}{2}\right)$$

그런 다음 사전 워핑된 주파수로 아날로그 필터를 설계하고 쌍일차 변환을 적용합니다.

### 8.5 단계별 BLT 설계

```python
def iir_design_blt(wp_digital, ws_digital, Rp_dB, As_dB, fs, ftype='butter'):
    """
    Complete IIR design using bilinear transform with pre-warping.

    Parameters:
        wp_digital: digital passband edge (Hz)
        ws_digital: digital stopband edge (Hz)
        Rp_dB: passband ripple (dB)
        As_dB: stopband attenuation (dB)
        fs: sampling frequency (Hz)
        ftype: 'butter', 'cheby1', 'cheby2', or 'ellip'

    Returns:
        b, a: digital filter coefficients
        N: filter order
    """
    T = 1 / fs

    # Step 1: Pre-warp digital frequencies to analog
    Omega_p = 2 * fs * np.tan(np.pi * wp_digital / fs)
    Omega_s = 2 * fs * np.tan(np.pi * ws_digital / fs)

    print(f"Pre-warped frequencies: Ωp = {Omega_p:.2f}, Ωs = {Omega_s:.2f} rad/s")

    # Step 2: Determine order
    if ftype == 'butter':
        N, Wn = signal.buttord(Omega_p, Omega_s, Rp_dB, As_dB, analog=True)
        # Step 3: Design analog prototype
        ba, aa = signal.butter(N, Wn, analog=True)
    elif ftype == 'cheby1':
        N, Wn = signal.cheb1ord(Omega_p, Omega_s, Rp_dB, As_dB, analog=True)
        ba, aa = signal.cheby1(N, Rp_dB, Wn, analog=True)
    elif ftype == 'cheby2':
        N, Wn = signal.cheb2ord(Omega_p, Omega_s, Rp_dB, As_dB, analog=True)
        ba, aa = signal.cheby2(N, As_dB, Wn, analog=True)
    elif ftype == 'ellip':
        N, Wn = signal.ellipord(Omega_p, Omega_s, Rp_dB, As_dB, analog=True)
        ba, aa = signal.ellip(N, Rp_dB, As_dB, Wn, analog=True)

    print(f"Analog filter order: N = {N}")

    # Step 4: Bilinear transform
    b, a = signal.bilinear(ba, aa, fs)

    return b, a, N

# Example
fs = 8000
wp = 1000  # Hz
ws = 1500  # Hz
Rp = 1.0   # dB
As = 60    # dB

b, a, N = iir_design_blt(wp, ws, Rp, As, fs, ftype='ellip')
print(f"\nDigital filter coefficients:")
print(f"b = {b}")
print(f"a = {a}")

# Verify
w, H = signal.freqz(b, a, worN=4096, fs=fs)
H_dB = 20 * np.log10(np.abs(H) + 1e-15)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Magnitude
axes[0, 0].plot(w, H_dB, 'b-', linewidth=1.5)
axes[0, 0].axhline(-Rp, color='g', linestyle='--', alpha=0.5, label=f'-{Rp} dB')
axes[0, 0].axhline(-As, color='r', linestyle='--', alpha=0.5, label=f'-{As} dB')
axes[0, 0].axvline(wp, color='g', linestyle=':', alpha=0.5)
axes[0, 0].axvline(ws, color='r', linestyle=':', alpha=0.5)
axes[0, 0].set_xlabel('Frequency (Hz)')
axes[0, 0].set_ylabel('Magnitude (dB)')
axes[0, 0].set_title(f'Elliptic IIR (N={N}): Magnitude')
axes[0, 0].set_ylim(-80, 5)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Phase
axes[0, 1].plot(w, np.unwrap(np.angle(H)) * 180 / np.pi, 'r-', linewidth=1.5)
axes[0, 1].set_xlabel('Frequency (Hz)')
axes[0, 1].set_ylabel('Phase (degrees)')
axes[0, 1].set_title('Phase Response')
axes[0, 1].grid(True, alpha=0.3)

# Group delay
w_gd, gd = signal.group_delay((b, a), w=4096, fs=fs)
axes[1, 0].plot(w_gd, gd, 'g-', linewidth=1.5)
axes[1, 0].set_xlabel('Frequency (Hz)')
axes[1, 0].set_ylabel('Group Delay (samples)')
axes[1, 0].set_title('Group Delay (Non-constant!)')
axes[1, 0].grid(True, alpha=0.3)

# Pole-zero plot
z_zeros, z_poles, k = signal.tf2zpk(b, a)
theta = np.linspace(0, 2 * np.pi, 200)
axes[1, 1].plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
axes[1, 1].plot(np.real(z_zeros), np.imag(z_zeros), 'bo', markersize=10, label='Zeros')
axes[1, 1].plot(np.real(z_poles), np.imag(z_poles), 'rx', markersize=10,
                markeredgewidth=2, label='Poles')
axes[1, 1].set_xlabel('Real')
axes[1, 1].set_ylabel('Imaginary')
axes[1, 1].set_title('Pole-Zero Plot')
axes[1, 1].set_aspect('equal')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('IIR Filter Design via Bilinear Transform', fontsize=13)
plt.tight_layout()
plt.savefig('iir_blt_design.png', dpi=150)
plt.close()
```

---

## 9. 임펄스 불변 방법

### 9.1 원리

임펄스 불변(impulse invariance) 방법은 디지털 필터의 임펄스 응답을 아날로그 임펄스 응답의 샘플과 동일하게 설정합니다:

$$h[n] = T \cdot h_a(nT)$$

여기서 $h_a(t)$는 아날로그 임펄스 응답, $T$는 샘플링 주기입니다.

### 9.2 매핑

아날로그 필터가 부분 분수 전개(partial fraction expansion)를 가질 때:

$$H_a(s) = \sum_{k=1}^{N} \frac{A_k}{s - s_k}$$

디지털 필터는:

$$H(z) = T \sum_{k=1}^{N} \frac{A_k}{1 - e^{s_k T} z^{-1}}$$

각 아날로그 극점 $s_k$는 디지털 극점 $z_k = e^{s_k T}$로 매핑됩니다.

### 9.3 에일리어싱 문제

디지털 필터의 주파수 응답은 아날로그 응답의 이동된 복사본들의 합입니다:

$$H(e^{j\omega}) = \frac{1}{T} \sum_{k=-\infty}^{\infty} H_a\left(j\frac{\omega - 2\pi k}{T}\right)$$

이로 인해 $H_a(j\Omega)$가 대역 제한(bandlimited)되지 않으면 **에일리어싱(aliasing)**이 발생하므로, 임펄스 불변 방법은 고역 통과 및 대역 저지 필터에 적합하지 않습니다.

### 9.4 구현 예제

```python
def impulse_invariance(ba, aa, fs):
    """
    Apply impulse invariance method to convert analog filter to digital.

    Parameters:
        ba, aa: analog filter coefficients (transfer function form)
        fs: sampling frequency

    Returns:
        bd, ad: digital filter coefficients
    """
    T = 1 / fs

    # Convert to zeros, poles, gain form
    z_a, p_a, k_a = signal.tf2zpk(ba, aa)

    # Partial fraction expansion
    residues, poles, direct = signal.residue(ba, aa)

    # Map poles: z_k = exp(s_k * T)
    z_poles = np.exp(poles * T)

    # Construct digital transfer function
    # H(z) = T * sum(A_k / (1 - e^(s_k*T) * z^-1))
    bd = np.array([0.0])
    ad = np.array([1.0])

    for A_k, z_k in zip(residues, z_poles):
        # Each term: T * A_k / (1 - z_k * z^-1)
        bd_k = np.array([T * A_k])
        ad_k = np.array([1, -z_k])

        # Combine fractions
        bd_new = np.convolve(bd, ad_k) + np.convolve(bd_k, ad)
        ad_new = np.convolve(ad, ad_k)
        bd = bd_new
        ad = ad_new

    # Ensure real coefficients
    bd = np.real(bd)
    ad = np.real(ad)

    return bd, ad

# Compare BLT and impulse invariance for a Butterworth lowpass
fs = 8000
N = 4
fc = 1000  # Cutoff frequency (Hz)

# Analog prototype
Omega_c = 2 * np.pi * fc
ba, aa = signal.butter(N, Omega_c, analog=True)

# Method 1: Bilinear transform
bd_blt, ad_blt = signal.bilinear(ba, aa, fs)

# Method 2: Impulse invariance
bd_ii, ad_ii = impulse_invariance(ba, aa, fs)

# Compare frequency responses
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

w_blt, H_blt = signal.freqz(bd_blt, ad_blt, worN=4096, fs=fs)
w_ii, H_ii = signal.freqz(bd_ii, ad_ii, worN=4096, fs=fs)

# Also plot the analog response for reference
w_a, H_a = signal.freqs(ba, aa, worN=np.linspace(0, 2 * np.pi * fs / 2, 4096))

axes[0].plot(w_a / (2 * np.pi), 20 * np.log10(np.abs(H_a) + 1e-15),
             'k--', linewidth=1.5, label='Analog', alpha=0.5)
axes[0].plot(w_blt, 20 * np.log10(np.abs(H_blt) + 1e-15),
             'b-', linewidth=1.5, label='Bilinear Transform')
axes[0].plot(w_ii, 20 * np.log10(np.abs(H_ii) + 1e-15),
             'r-', linewidth=1.5, label='Impulse Invariance')
axes[0].set_xlabel('Frequency (Hz)')
axes[0].set_ylabel('Magnitude (dB)')
axes[0].set_title('BLT vs Impulse Invariance')
axes[0].set_ylim(-80, 5)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Near Nyquist detail
axes[1].plot(w_blt, 20 * np.log10(np.abs(H_blt) + 1e-15),
             'b-', linewidth=1.5, label='BLT')
axes[1].plot(w_ii, 20 * np.log10(np.abs(H_ii) + 1e-15),
             'r-', linewidth=1.5, label='Impulse Invariance')
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Magnitude (dB)')
axes[1].set_title('Detail Near Nyquist (aliasing visible)')
axes[1].set_xlim(2000, 4000)
axes[1].set_ylim(-60, -20)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('impulse_invariance.png', dpi=150)
plt.close()
```

---

## 10. 완전한 IIR 설계 절차

### 10.1 단계별 절차

```
┌──────────────────────────────────────────────────────────────────┐
│                  IIR Filter Design Steps                         │
│                                                                  │
│  1. Specify digital requirements:                                │
│     - Passband/stopband edges (ωp, ωs or fp, fs)               │
│     - Passband ripple Rp (dB)                                    │
│     - Stopband attenuation As (dB)                               │
│                                                                  │
│  2. Choose filter type:                                          │
│     - Butterworth: maximally flat                                │
│     - Chebyshev I: equiripple passband                           │
│     - Chebyshev II: equiripple stopband                          │
│     - Elliptic: minimum order                                    │
│                                                                  │
│  3. Choose conversion method (usually BLT)                       │
│                                                                  │
│  4. Pre-warp critical frequencies                                │
│                                                                  │
│  5. Determine minimum analog filter order                        │
│                                                                  │
│  6. Design analog prototype                                      │
│                                                                  │
│  7. Apply analog-to-digital transformation                       │
│                                                                  │
│  8. Verify specifications                                        │
│     - Check passband ripple                                      │
│     - Check stopband attenuation                                 │
│     - Check stability (all poles inside unit circle)             │
│                                                                  │
│  9. Implement (Direct Form II, SOS cascade, etc.)                │
└──────────────────────────────────────────────────────────────────┘
```

### 10.2 scipy.signal 직접 설계 함수 사용

SciPy는 사전 워핑을 내부적으로 처리하는 고수준 함수를 제공합니다:

```python
def design_all_types(fs, f_pass, f_stop, Rp, As):
    """Design IIR lowpass filter using all four types."""
    results = {}

    # Direct digital design (scipy handles pre-warping internally)
    # Butterworth
    N_b, Wn_b = signal.buttord(f_pass, f_stop, Rp, As, fs=fs)
    b_b, a_b = signal.butter(N_b, Wn_b, fs=fs)
    results['Butterworth'] = (b_b, a_b, N_b)

    # Chebyshev Type I
    N_c1, Wn_c1 = signal.cheb1ord(f_pass, f_stop, Rp, As, fs=fs)
    b_c1, a_c1 = signal.cheby1(N_c1, Rp, Wn_c1, fs=fs)
    results['Chebyshev I'] = (b_c1, a_c1, N_c1)

    # Chebyshev Type II
    N_c2, Wn_c2 = signal.cheb2ord(f_pass, f_stop, Rp, As, fs=fs)
    b_c2, a_c2 = signal.cheby2(N_c2, As, Wn_c2, fs=fs)
    results['Chebyshev II'] = (b_c2, a_c2, N_c2)

    # Elliptic
    N_e, Wn_e = signal.ellipord(f_pass, f_stop, Rp, As, fs=fs)
    b_e, a_e = signal.ellip(N_e, Rp, As, Wn_e, fs=fs)
    results['Elliptic'] = (b_e, a_e, N_e)

    return results

# Design and compare
fs = 16000
f_pass = 2000
f_stop = 2500
Rp = 0.5
As = 60

results = design_all_types(fs, f_pass, f_stop, Rp, As)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, (name, (b, a, N)) in zip(axes.flat, results.items()):
    w, H = signal.freqz(b, a, worN=4096, fs=fs)
    H_dB = 20 * np.log10(np.abs(H) + 1e-15)

    ax.plot(w, H_dB, 'b-', linewidth=1.5)
    ax.axhline(-Rp, color='g', linestyle='--', alpha=0.5, label=f'Rp = -{Rp} dB')
    ax.axhline(-As, color='r', linestyle='--', alpha=0.5, label=f'As = -{As} dB')
    ax.axvline(f_pass, color='g', linestyle=':', alpha=0.5)
    ax.axvline(f_stop, color='r', linestyle=':', alpha=0.5)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title(f'{name} (N={N})')
    ax.set_ylim(-80, 5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle(f'IIR Filter Comparison (fp={f_pass} Hz, fs_stop={f_stop} Hz)', fontsize=13)
plt.tight_layout()
plt.savefig('iir_all_types.png', dpi=150)
plt.close()

# Print summary
print(f"\n{'Filter Type':<15} {'Order':<8} {'Actual Rp (dB)':<16} {'Actual As (dB)':<16}")
print("-" * 55)
for name, (b, a, N) in results.items():
    w, H = signal.freqz(b, a, worN=4096, fs=fs)
    H_mag = np.abs(H)
    f = w

    # Passband ripple
    pass_idx = f <= f_pass
    Rp_actual = -20 * np.log10(np.min(H_mag[pass_idx]))

    # Stopband attenuation
    stop_idx = f >= f_stop
    As_actual = -20 * np.log10(np.max(H_mag[stop_idx]))

    print(f"{name:<15} {N:<8} {Rp_actual:<16.4f} {As_actual:<16.4f}")
```

---

## 11. 안정성 분석

### 11.1 안정성 기준

디지털 IIR 필터는 $H(z)$의 모든 극점이 단위원 내부에 엄격히 위치할 때만 **안정(stable)**합니다:

$$|z_k| < 1, \quad \forall k$$

### 11.2 수치적 안정성을 위한 2차 섹션(SOS)

직접 형식(direct form)으로 구현된 고차 IIR 필터는 계수 양자화(coefficient quantization) 및 수치 정밀도 문제가 발생할 수 있습니다. 해결책은 **2차 섹션(second-order sections, biquads)의 종속 연결(cascade)**로 구현하는 것입니다:

$$H(z) = \prod_{i=1}^{L} \frac{b_{0i} + b_{1i}z^{-1} + b_{2i}z^{-2}}{1 + a_{1i}z^{-1} + a_{2i}z^{-2}}$$

```python
def stability_analysis(b, a, title=""):
    """Analyze IIR filter stability."""
    # Poles and zeros
    zeros, poles, gain = signal.tf2zpk(b, a)

    # Check stability
    pole_mags = np.abs(poles)
    is_stable = np.all(pole_mags < 1.0)
    max_pole_mag = np.max(pole_mags) if len(poles) > 0 else 0

    print(f"Filter: {title}")
    print(f"  Number of poles: {len(poles)}")
    print(f"  Number of zeros: {len(zeros)}")
    print(f"  Maximum pole magnitude: {max_pole_mag:.6f}")
    print(f"  Stable: {is_stable}")
    print(f"  Stability margin: {1.0 - max_pole_mag:.6f}")

    # Pole-zero plot
    fig, ax = plt.subplots(figsize=(8, 8))
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', alpha=0.3, label='Unit circle')
    ax.plot(np.real(zeros), np.imag(zeros), 'bo', markersize=10,
            label=f'Zeros ({len(zeros)})')
    ax.plot(np.real(poles), np.imag(poles), 'rx', markersize=10,
            markeredgewidth=2, label=f'Poles ({len(poles)})')

    # Color poles by stability
    for p in poles:
        color = 'green' if np.abs(p) < 1 else 'red'
        circle = plt.Circle((np.real(p), np.imag(p)), 0.03,
                            fill=True, color=color, alpha=0.3)
        ax.add_patch(circle)

    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title(f'Pole-Zero Plot: {title}\n(Stable: {is_stable}, '
                 f'Max |pole| = {max_pole_mag:.4f})')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    plt.tight_layout()
    return fig, is_stable

# Design a high-order filter and check stability
N = 12
b_high, a_high = signal.butter(N, 0.3)  # 12th-order Butterworth
fig, stable = stability_analysis(b_high, a_high, f"Butterworth N={N}")
plt.savefig('stability_analysis.png', dpi=150)
plt.close()

# Convert to SOS form for better numerical stability
sos = signal.tf2sos(b_high, a_high)
print(f"\nSOS representation: {sos.shape[0]} second-order sections")
print(f"SOS coefficients:\n{sos}")
```

### 11.3 수치적 비교: tf vs SOS

```python
def compare_tf_vs_sos(order=20):
    """Compare direct form vs SOS implementation for high-order filter."""
    # Design high-order filter
    b, a = signal.butter(order, 0.1)
    sos = signal.butter(order, 0.1, output='sos')

    # Test signal
    np.random.seed(42)
    x = np.random.randn(1000)

    # Filter using tf form
    y_tf = signal.lfilter(b, a, x)

    # Filter using SOS form
    y_sos = signal.sosfilt(sos, x)

    # Compare
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Frequency response comparison
    w_tf, H_tf = signal.freqz(b, a, worN=4096)
    w_sos, H_sos = signal.sosfreqz(sos, worN=4096)

    axes[0, 0].plot(w_tf / np.pi, 20 * np.log10(np.abs(H_tf) + 1e-15),
                     'b-', linewidth=1.5, label='tf form')
    axes[0, 0].plot(w_sos / np.pi, 20 * np.log10(np.abs(H_sos) + 1e-15),
                     'r--', linewidth=1.5, label='SOS form')
    axes[0, 0].set_xlabel('Normalized Frequency (×π)')
    axes[0, 0].set_ylabel('Magnitude (dB)')
    axes[0, 0].set_title('Frequency Response')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Output comparison
    axes[0, 1].plot(y_tf[:200], 'b-', alpha=0.7, label='tf form')
    axes[0, 1].plot(y_sos[:200], 'r--', alpha=0.7, label='SOS form')
    axes[0, 1].set_xlabel('Sample')
    axes[0, 1].set_ylabel('Output')
    axes[0, 1].set_title('Output Signal')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Difference
    diff = np.abs(y_tf - y_sos)
    axes[1, 0].semilogy(diff, 'k-')
    axes[1, 0].set_xlabel('Sample')
    axes[1, 0].set_ylabel('|y_tf - y_sos|')
    axes[1, 0].set_title(f'Numerical Difference (max: {np.max(diff):.2e})')
    axes[1, 0].grid(True, alpha=0.3)

    # Pole magnitudes
    zeros_tf, poles_tf, _ = signal.tf2zpk(b, a)
    axes[1, 1].plot(np.abs(poles_tf), 'rx', markersize=10, markeredgewidth=2)
    axes[1, 1].axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Unit circle')
    axes[1, 1].set_xlabel('Pole Index')
    axes[1, 1].set_ylabel('|pole|')
    axes[1, 1].set_title(f'Pole Magnitudes (N={order})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'Direct Form vs SOS (Order={order})', fontsize=13)
    plt.tight_layout()
    plt.savefig('tf_vs_sos.png', dpi=150)
    plt.close()

compare_tf_vs_sos(order=20)
```

---

## 12. 필터 유형 비교

### 12.1 요약 표

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     IIR Filter Type Comparison                          │
├──────────────┬─────────────┬────────────┬────────────┬────────────────┤
│              │ Butterworth │ Chebyshev I│ Chebyshev II│ Elliptic      │
├──────────────┼─────────────┼────────────┼────────────┼────────────────┤
│ Passband     │ Maximally   │ Equiripple │ Monotonic  │ Equiripple    │
│              │ flat        │            │            │               │
├──────────────┼─────────────┼────────────┼────────────┼────────────────┤
│ Stopband     │ Monotonic   │ Monotonic  │ Equiripple │ Equiripple    │
├──────────────┼─────────────┼────────────┼────────────┼────────────────┤
│ Transition   │ Widest      │ Medium     │ Medium     │ Sharpest      │
│ band         │             │            │            │               │
├──────────────┼─────────────┼────────────┼────────────┼────────────────┤
│ Order for    │ Highest     │ Medium     │ Medium     │ Lowest        │
│ given specs  │             │            │            │               │
├──────────────┼─────────────┼────────────┼────────────┼────────────────┤
│ Group delay  │ Most        │ Less       │ Better than│ Most          │
│              │ uniform     │ uniform    │ Type I     │ non-uniform   │
├──────────────┼─────────────┼────────────┼────────────┼────────────────┤
│ Use case     │ General,    │ Sharp      │ Flat       │ Minimum order,│
│              │ smooth      │ cutoff,    │ passband,  │ tight specs   │
│              │ response    │ ok ripple  │ ok stopband│               │
└──────────────┴─────────────┴────────────┴────────────┴────────────────┘
```

### 12.2 종합 시각적 비교

```python
def comprehensive_comparison():
    """Complete comparison of all four IIR filter types."""
    fs = 16000
    fp = 2000
    fstop = 3000
    Rp = 1.0
    As = 50

    filters = {}

    # Butterworth
    N, Wn = signal.buttord(fp, fstop, Rp, As, fs=fs)
    sos = signal.butter(N, Wn, fs=fs, output='sos')
    filters['Butterworth'] = (sos, N)

    # Chebyshev I
    N, Wn = signal.cheb1ord(fp, fstop, Rp, As, fs=fs)
    sos = signal.cheby1(N, Rp, Wn, fs=fs, output='sos')
    filters['Chebyshev I'] = (sos, N)

    # Chebyshev II
    N, Wn = signal.cheb2ord(fp, fstop, Rp, As, fs=fs)
    sos = signal.cheby2(N, As, Wn, fs=fs, output='sos')
    filters['Chebyshev II'] = (sos, N)

    # Elliptic
    N, Wn = signal.ellipord(fp, fstop, Rp, As, fs=fs)
    sos = signal.ellip(N, Rp, As, Wn, fs=fs, output='sos')
    filters['Elliptic'] = (sos, N)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {'Butterworth': 'blue', 'Chebyshev I': 'red',
              'Chebyshev II': 'green', 'Elliptic': 'purple'}

    # Magnitude response overlay
    ax = axes[0, 0]
    for name, (sos, N) in filters.items():
        w, H = signal.sosfreqz(sos, worN=4096, fs=fs)
        ax.plot(w, 20 * np.log10(np.abs(H) + 1e-15),
                color=colors[name], linewidth=1.5, label=f'{name} (N={N})')
    ax.axhline(-Rp, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(-As, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Magnitude Response')
    ax.set_ylim(-70, 5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Passband detail
    ax = axes[0, 1]
    for name, (sos, N) in filters.items():
        w, H = signal.sosfreqz(sos, worN=4096, fs=fs)
        ax.plot(w, 20 * np.log10(np.abs(H) + 1e-15),
                color=colors[name], linewidth=1.5, label=f'{name} (N={N})')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Passband Detail')
    ax.set_xlim(0, fp * 1.2)
    ax.set_ylim(-2, 0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Group delay
    ax = axes[1, 0]
    for name, (sos, N) in filters.items():
        b, a = signal.sos2tf(sos)
        w_gd, gd = signal.group_delay((b, a), w=4096, fs=fs)
        ax.plot(w_gd, gd, color=colors[name], linewidth=1.5, label=f'{name}')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Group Delay (samples)')
    ax.set_title('Group Delay')
    ax.set_xlim(0, fs / 2)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Step response
    ax = axes[1, 1]
    step = np.ones(200)
    for name, (sos, N) in filters.items():
        y = signal.sosfilt(sos, step)
        ax.plot(y, color=colors[name], linewidth=1.5, label=f'{name}')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')
    ax.set_title('Step Response')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Comprehensive IIR Filter Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig('comprehensive_comparison.png', dpi=150)
    plt.close()

comprehensive_comparison()
```

---

## 13. Python 구현

### 13.1 완전한 IIR 설계 워크플로우

```python
def complete_iir_workflow(fs, filter_type, band_type, freqs, Rp, As):
    """
    Complete IIR filter design and analysis workflow.

    Parameters:
        fs: sampling frequency (Hz)
        filter_type: 'butter', 'cheby1', 'cheby2', 'ellip'
        band_type: 'lowpass', 'highpass', 'bandpass', 'bandstop'
        freqs: (f_pass, f_stop) or ((fp1, fp2), (fs1, fs2)) for bandpass/stop
        Rp: passband ripple (dB)
        As: stopband attenuation (dB)

    Returns:
        sos: second-order sections
        info: design information
    """
    f_pass, f_stop = freqs

    # Order determination
    ord_funcs = {
        'butter': signal.buttord,
        'cheby1': signal.cheb1ord,
        'cheby2': signal.cheb2ord,
        'ellip': signal.ellipord,
    }
    N, Wn = ord_funcs[filter_type](f_pass, f_stop, Rp, As, fs=fs)

    # Design
    design_funcs = {
        'butter': lambda: signal.butter(N, Wn, btype=band_type, fs=fs, output='sos'),
        'cheby1': lambda: signal.cheby1(N, Rp, Wn, btype=band_type, fs=fs, output='sos'),
        'cheby2': lambda: signal.cheby2(N, As, Wn, btype=band_type, fs=fs, output='sos'),
        'ellip': lambda: signal.ellip(N, Rp, As, Wn, btype=band_type, fs=fs, output='sos'),
    }
    sos = design_funcs[filter_type]()

    info = {
        'filter_type': filter_type,
        'band_type': band_type,
        'order': N,
        'Wn': Wn,
        'Rp': Rp,
        'As': As,
    }

    return sos, info

# Example: Bandpass filter
fs = 44100
sos_bp, info_bp = complete_iir_workflow(
    fs=fs,
    filter_type='ellip',
    band_type='bandpass',
    freqs=([800, 3000], [500, 3500]),
    Rp=0.5,
    As=50
)

print(f"Bandpass filter: {info_bp}")

# Apply to signal
t = np.arange(0, 0.5, 1/fs)
# Speech-like signal: mix of tones + noise
x = (0.5 * np.sin(2*np.pi*200*t) +    # Low freq
     np.sin(2*np.pi*1000*t) +           # Mid freq (passband)
     0.8 * np.sin(2*np.pi*2000*t) +     # Mid freq (passband)
     0.3 * np.sin(2*np.pi*5000*t) +     # High freq
     0.2 * np.random.randn(len(t)))      # Noise

y = signal.sosfilt(sos_bp, x)

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Time domain
axes[0].plot(t[:2000] * 1000, x[:2000], 'b-', alpha=0.7, label='Input')
axes[0].plot(t[:2000] * 1000, y[:2000], 'r-', linewidth=1.5, label='Filtered')
axes[0].set_xlabel('Time (ms)')
axes[0].set_ylabel('Amplitude')
axes[0].set_title('Time Domain: Bandpass Filtering')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Spectrum
N_fft = len(x)
freqs_fft = np.fft.rfftfreq(N_fft, 1/fs)
X = np.abs(np.fft.rfft(x)) / N_fft
Y = np.abs(np.fft.rfft(y)) / N_fft

axes[1].plot(freqs_fft, 20*np.log10(X + 1e-15), 'b-', alpha=0.7, label='Input')
axes[1].plot(freqs_fft, 20*np.log10(Y + 1e-15), 'r-', linewidth=1.5, label='Filtered')
axes[1].axvspan(800, 3000, alpha=0.1, color='green', label='Passband')
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Magnitude (dB)')
axes[1].set_title('Frequency Domain')
axes[1].set_xlim(0, 8000)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Filter frequency response
w, H = signal.sosfreqz(sos_bp, worN=4096, fs=fs)
axes[2].plot(w, 20*np.log10(np.abs(H) + 1e-15), 'b-', linewidth=1.5)
axes[2].axvspan(800, 3000, alpha=0.1, color='green', label='Passband')
axes[2].set_xlabel('Frequency (Hz)')
axes[2].set_ylabel('Magnitude (dB)')
axes[2].set_title(f'Filter Response (Elliptic N={info_bp["order"]})')
axes[2].set_xlim(0, 8000)
axes[2].set_ylim(-70, 5)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('iir_bandpass_demo.png', dpi=150)
plt.close()
```

### 13.2 SOS를 사용한 실시간 IIR 필터링

```python
class RealtimeIIRFilter:
    """Real-time IIR filter using second-order sections."""

    def __init__(self, sos):
        """
        Initialize with SOS coefficients.

        Parameters:
            sos: second-order sections array (L x 6)
        """
        self.sos = np.array(sos, dtype=np.float64)
        self.n_sections = self.sos.shape[0]
        # State variables: 2 delay elements per section
        self.state = np.zeros((self.n_sections, 2))

    def process_sample(self, x):
        """Process a single input sample."""
        for i in range(self.n_sections):
            b0, b1, b2, a0, a1, a2 = self.sos[i]

            # Direct Form II Transposed
            y = b0 * x + self.state[i, 0]
            self.state[i, 0] = b1 * x - a1 * y + self.state[i, 1]
            self.state[i, 1] = b2 * x - a2 * y

            x = y  # Output of this section is input to next

        return y

    def process_block(self, x_block):
        """Process a block of samples."""
        y = np.zeros_like(x_block)
        for n in range(len(x_block)):
            y[n] = self.process_sample(x_block[n])
        return y

    def reset(self):
        """Reset filter state."""
        self.state = np.zeros((self.n_sections, 2))

# Demonstration
sos = signal.butter(6, 1000, fs=8000, output='sos')
filt = RealtimeIIRFilter(sos)

# Process in blocks (simulating real-time)
np.random.seed(42)
x = np.random.randn(1000)
block_size = 64

y_realtime = np.zeros_like(x)
for i in range(0, len(x), block_size):
    block = x[i:i + block_size]
    y_realtime[i:i + len(block)] = filt.process_block(block)

# Compare with scipy batch processing
y_batch = signal.sosfilt(sos, x)

error = np.max(np.abs(y_realtime - y_batch))
print(f"Maximum error between real-time and batch: {error:.2e}")
```

---

## 14. 연습 문제

### 연습 1: 버터워스 필터 설계

다음 규격의 디지털 버터워스 저역 통과 필터를 설계하세요:
- 샘플링 주파수: $f_s = 10000$ Hz
- 통과대역 엣지: $f_p = 1500$ Hz, 리플 $\leq 0.5$ dB
- 저지대역 엣지: $f_s = 2000$ Hz, 감쇠 $\geq 40$ dB

(a) 버터워스 차수 공식을 사용하여 필요한 필터 차수를 계산하세요.

(b) 디지털 주파수를 아날로그 주파수로 사전 워핑하세요.

(c) 아날로그 버터워스 원형을 설계하고 쌍일차 변환을 적용하세요.

(d) 설계가 규격을 충족하는지 검증하세요. 크기 응답, 위상 응답, 군 지연(group delay), 극-영점 다이어그램을 그리세요.

(e) 수동 설계를 `scipy.signal.butter`와 비교하세요. 결과가 동일한가요?

### 연습 2: 체비쇼프 필터 비교

규격: $f_s = 8000$ Hz, $f_p = 1000$ Hz, $f_\text{stop} = 1200$ Hz, $R_p = 1$ dB, $A_s = 50$ dB에 대해:

(a) 체비쇼프 Type I과 Type II 필터를 모두 설계하세요.

(b) 단일 그래프에서 주파수 응답을 비교하세요.

(c) 통과대역에서 어느 것이 군 지연 평탄도가 더 좋은가요? 정량적으로 보여주세요.

(d) 두 필터를 처프(chirp) 신호(100 Hz에서 4000 Hz 스윕)에 적용하고 출력을 비교하세요.

### 연습 3: 오디오용 타원형 필터

오디오 안티에일리어싱(anti-aliasing)을 위한 타원형 저역 통과 필터를 설계하세요:
- 입력 샘플링 속도: 96 kHz (48 kHz로 다운샘플링 예정)
- 통과대역: 0~20 kHz, 리플 $\leq 0.01$ dB
- 저지대역: 24 kHz~48 kHz, 감쇠 $\geq 96$ dB (16비트 정밀도)

(a) 최소 필터 차수를 결정하세요.

(b) SOS 형식으로 구현하세요.

(c) 선형 및 dB 스케일 모두에서 크기 응답을 그리세요.

(d) 군 지연을 계산하고 그리세요. 오디오에서 군 지연 변동이 허용 가능한가요?

### 연습 4: 수작업 쌍일차 변환

1차 아날로그 저역 통과 필터가 주어졌을 때:

$$H_a(s) = \frac{\Omega_c}{s + \Omega_c}$$

(a) 쌍일차 변환 $s = \frac{2}{T}\frac{z-1}{z+1}$을 적용하여 $H(z)$를 유도하세요.

(b) $\Omega_c = 2\pi \times 1000$ rad/s, $f_s = 8000$ Hz에 대해 디지털 필터 계수 $b_0, b_1, a_0, a_1$을 계산하세요.

(c) $H(z)$의 주파수 응답과 워핑된 아날로그 응답을 비교하여 검증하세요.

(d) 동일한 차단 주파수의 2차 버터워스 필터에 대해 반복하세요.

### 연습 5: 안정성 조사

(a) 16차 체비쇼프 Type I 필터($R_p = 3$ dB, $\omega_c = 0.1\pi$)를 `tf`와 `sos` 형식으로 설계하세요.

(b) 두 표현에 대해 극점 크기를 계산하고 표시하세요.

(c) 두 표현을 사용하여 백색 잡음 신호를 필터링하세요. `tf` 형식 출력이 유효한가요?

(d) 체비쇼프 Type I 필터에서 `tf` 형식이 일반적으로 어느 차수에서 무너지기 시작하나요? 차수 4, 8, 12, 16, 20, 24로 실험해보세요.

### 연습 6: IIR vs FIR 비교

규격: $f_s = 16000$ Hz, $f_p = 3000$ Hz, $f_\text{stop} = 3500$ Hz, $A_s = 60$ dB에 대해:

(a) IIR(타원형)과 FIR(Parks-McClellan) 필터 모두 설계하세요.

(b) 다음을 비교하세요: 필터 차수, 샘플당 계산 비용, 군 지연, 크기 응답.

(c) 음성과 유사한 신호(100, 200, ..., 5000 Hz의 고조파 합산)를 필터링하세요. 시간 영역 출력을 시각적으로 비교하고 SNR 개선을 계산하세요.

(d) IIR을 FIR보다 선택해야 하는 응용 시나리오는 무엇인지, 그 반대는 무엇인지 설명하세요?

### 연습 7: 대역 저지(노치) 필터

심전도(ECG) 신호에서 60 Hz 전원선 간섭(powerline interference)을 제거하기 위한 좁은 대역 저지 필터를 설계하세요:
- 샘플링 주파수: 500 Hz
- 제거 대상: 59-61 Hz
- 통과대역: 0-55 Hz 및 65-250 Hz
- 통과대역 리플: $\leq 0.1$ dB
- 저지대역 감쇠: $\geq 30$ dB

(a) 타원형 원형을 사용하여 설계하세요. 어떤 차수가 필요한가요?

(b) 60 Hz 주변을 확대한 뷰로 크기 응답을 그리세요.

(c) 60 Hz 오염이 있는 합성 ECG 신호를 생성하고 필터의 효과를 시연하세요.

(d) 버터워스 대 타원형 구현의 군 지연을 계산하고 비교하세요.

---

## 참고문헌

1. **Oppenheim, A. V., & Schafer, R. W. (2010).** *Discrete-Time Signal Processing* (3rd ed.). Pearson. Chapter 7.
2. **Proakis, J. G., & Manolakis, D. G. (2007).** *Digital Signal Processing* (4th ed.). Pearson. Chapter 11.
3. **Parks, T. W., & Burrus, C. S. (1987).** *Digital Filter Design*. Wiley.
4. **Antoniou, A. (2006).** *Digital Signal Processing: Signals, Systems, and Filters*. McGraw-Hill.
5. **SciPy Documentation** -- Filter design functions: https://docs.scipy.org/doc/scipy/reference/signal.html
6. **Smith, S. W. (1997).** *The Scientist and Engineer's Guide to Digital Signal Processing*. Available free at http://www.dspguide.com/

---

## 탐색

- 이전: [09. FIR 필터 설계](09_FIR_Filter_Design.md)
- 다음: [11. 다중률 신호 처리](11_Multirate_Processing.md)
- [목차로 돌아가기](00_Overview.md)
