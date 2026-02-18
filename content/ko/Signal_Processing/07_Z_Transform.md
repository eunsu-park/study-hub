# Z 변환(Z-Transform)

## 개요

Z 변환은 라플라스 변환(Laplace Transform)의 이산 시간(discrete-time) 대응 개념입니다. 차분 방정식(difference equation)을 대수 방정식으로 변환하여, 복소 $z$ 평면에서 이산 시간 LTI 시스템을 분석할 수 있게 해줍니다. Z 변환은 시스템 안정성, 주파수 응답, 전달 함수 결정에 강력한 도구를 제공합니다. 이 레슨에서는 Z 변환 이론, 성질, 역변환 방법, 그리고 디지털 시스템 분석 응용을 다룹니다.

**학습 목표:**
- 양방향(bilateral) 및 단방향(unilateral) Z 변환을 정의하고 계산하기
- 수렴 영역(ROC, Region of Convergence)과 그 의미 이해하기
- 시스템 분석을 위한 Z 변환 성질 적용하기
- 다양한 방법으로 역 Z 변환 계산하기
- 전달 함수, 극점(poles), 영점(zeros)을 이용하여 LTI 시스템 분석하기
- Z 변환과 DTFT 및 라플라스 변환의 관계 이해하기

**선수 학습:** [06. 이산 푸리에 변환](06_Discrete_Fourier_Transform.md)

---

## 1. Z 변환의 정의

### 1.1 양방향 Z 변환(Bilateral Z-Transform)

이산 시간 신호 $x[n]$의 양방향(두 방향) Z 변환은 다음과 같습니다:

$$\boxed{X(z) = \mathcal{Z}\{x[n]\} = \sum_{n=-\infty}^{\infty} x[n] \, z^{-n}}$$

여기서 $z$는 복소 변수로, $z = r \, e^{j\omega}$입니다.

### 1.2 단방향 Z 변환(Unilateral Z-Transform)

단방향(한 방향) Z 변환은 초기 조건이 있는 인과 신호와 시스템에 사용됩니다:

$$X(z) = \sum_{n=0}^{\infty} x[n] \, z^{-n}$$

이 형태는 특히 비영(non-zero) 초기 조건이 있는 차분 방정식을 풀 때 유용합니다.

### 1.3 직관적 이해: z는 무엇인가?

복소 변수 $z = r \, e^{j\omega}$는 다음과 같이 분해됩니다:
- $|z| = r$: 원점으로부터의 반경(지수 가중을 통해 수렴 제어)
- $\angle z = \omega$: 각도(주파수에 대응)
- 단위원(unit circle) 위($r = 1$, $z = e^{j\omega}$): Z 변환은 DTFT로 축약됨

### 1.4 주요 Z 변환 쌍(Common Z-Transform Pairs)

| 신호 $x[n]$ | Z 변환 $X(z)$ | ROC |
|-------------|------------------|-----|
| $\delta[n]$ | $1$ | 모든 $z$ |
| $u[n]$ (단위 계단) | $\frac{z}{z-1} = \frac{1}{1-z^{-1}}$ | $|z| > 1$ |
| $a^n u[n]$ | $\frac{z}{z-a} = \frac{1}{1-az^{-1}}$ | $|z| > |a|$ |
| $-a^n u[-n-1]$ | $\frac{z}{z-a} = \frac{1}{1-az^{-1}}$ | $|z| < |a|$ |
| $n a^n u[n]$ | $\frac{az}{(z-a)^2} = \frac{az^{-1}}{(1-az^{-1})^2}$ | $|z| > |a|$ |
| $\cos(\omega_0 n) u[n]$ | $\frac{z(z-\cos\omega_0)}{z^2 - 2z\cos\omega_0 + 1}$ | $|z| > 1$ |
| $r^n \cos(\omega_0 n) u[n]$ | $\frac{z(z - r\cos\omega_0)}{z^2 - 2rz\cos\omega_0 + r^2}$ | $|z| > r$ |

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def compute_z_transform_examples():
    """Compute and verify common Z-transform pairs."""
    # Example 1: x[n] = (0.8)^n * u[n]
    a = 0.8
    N = 50
    n = np.arange(N)
    x = a ** n  # Causal exponential

    # Z-transform: X(z) = 1 / (1 - 0.8 * z^{-1}), |z| > 0.8
    # Evaluate on unit circle (should give DTFT)
    omega = np.linspace(-np.pi, np.pi, 1024)
    z_unit = np.exp(1j * omega)

    # X(z) on unit circle
    X_formula = 1.0 / (1.0 - a * z_unit ** (-1))

    # DTFT (direct computation from samples)
    X_dtft = np.zeros(len(omega), dtype=complex)
    for k in range(N):
        X_dtft += x[k] * np.exp(-1j * omega * k)

    # Compare
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(omega / np.pi, np.abs(X_formula), 'b-', linewidth=2,
                 label='Z-transform formula')
    axes[0].plot(omega / np.pi, np.abs(X_dtft), 'r--', linewidth=1,
                 label=f'DTFT (N={N} terms)')
    axes[0].set_title(r'$x[n] = 0.8^n u[n]$: Magnitude on Unit Circle')
    axes[0].set_xlabel(r'$\omega / \pi$')
    axes[0].set_ylabel(r'$|X(e^{j\omega})|$')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(omega / np.pi, np.angle(X_formula), 'b-', linewidth=2,
                 label='Z-transform formula')
    axes[1].plot(omega / np.pi, np.angle(X_dtft), 'r--', linewidth=1,
                 label=f'DTFT (N={N} terms)')
    axes[1].set_title('Phase on Unit Circle')
    axes[1].set_xlabel(r'$\omega / \pi$')
    axes[1].set_ylabel('Phase (radians)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ztransform_unit_circle.png', dpi=150)
    plt.show()

compute_z_transform_examples()
```

---

## 2. 수렴 영역(ROC, Region of Convergence)

### 2.1 정의

ROC는 Z 변환의 합이 수렴하는 모든 $z$ 값의 집합입니다:

$$\text{ROC} = \left\{ z \in \mathbb{C} : \sum_{n=-\infty}^{\infty} |x[n]| \, |z|^{-n} < \infty \right\}$$

ROC는 항상 z 평면에서 환형(annular) 영역입니다(원점을 중심으로 한 두 동심원 사이의 고리):

$$R^{-} < |z| < R^{+}$$

### 2.2 ROC 성질

1. **ROC에는 $X(z)$의 극점(poles)이 포함되지 않음**
2. **유한 지속 신호(finite-duration signals)**: ROC는 전체 z 평면($z = 0$ 및/또는 $z = \infty$ 제외 가능)
3. **오른쪽 신호(right-sided signals)** ($x[n] = 0$, $n < N_1$): ROC는 원 외부: $|z| > R^{-}$
4. **왼쪽 신호(left-sided signals)** ($x[n] = 0$, $n > N_2$): ROC는 원 내부: $|z| < R^{+}$
5. **양방향 신호(two-sided signals)**: ROC는 환형 고리
6. **DTFT 존재**: ROC가 단위원 $|z| = 1$을 포함할 경우에만 존재
7. **인과적이고 안정적인 시스템**: ROC가 단위원 외부까지 포함; 모든 극점은 단위원 내부에 위치

### 2.3 ROC와 신호 유형: 동일한 X(z), 다른 신호

Z 변환 $X(z) = \frac{1}{1 - az^{-1}}$은 ROC에 따라 **두 가지 다른 신호**에 대응할 수 있습니다:

- ROC: $|z| > |a|$ $\implies$ $x[n] = a^n u[n]$ (인과적, 오른쪽 신호)
- ROC: $|z| < |a|$ $\implies$ $x[n] = -a^n u[-n-1]$ (반인과적, 왼쪽 신호)

> 이 때문에 ROC가 필수적입니다: ROC 없이 $X(z)$ 단독으로는 $x[n]$을 유일하게 결정할 수 없습니다.

```python
def visualize_roc():
    """Visualize ROC for different signal types."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    theta = np.linspace(0, 2 * np.pi, 200)

    # Case 1: Causal signal x[n] = 0.7^n u[n], ROC: |z| > 0.7
    ax = axes[0]
    # Unit circle
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1)
    # Pole at z = 0.7
    ax.plot(0.7, 0, 'rx', markersize=12, markeredgewidth=2, label='Pole')
    # ROC: |z| > 0.7 (shade exterior)
    r_roc = 0.7
    circle = plt.Circle((0, 0), r_roc, fill=True, color='lightblue',
                         alpha=0.5, label=f'ROC: |z| > {r_roc}')
    ax.add_patch(circle)
    ax.fill_between(np.cos(theta) * 2, np.sin(theta) * 2,
                    alpha=0.2, color='green')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.set_title(r'Causal: $0.7^n u[n]$' + '\nROC: |z| > 0.7')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Case 2: Anti-causal signal, ROC: |z| < 0.7
    ax = axes[1]
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1)
    ax.plot(0.7, 0, 'rx', markersize=12, markeredgewidth=2, label='Pole')
    circle = plt.Circle((0, 0), r_roc, fill=True, color='lightgreen',
                         alpha=0.5, label=f'ROC: |z| < {r_roc}')
    ax.add_patch(circle)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.set_title(r'Anti-causal: $-0.7^n u[-n-1]$' + '\nROC: |z| < 0.7')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Case 3: Two-sided signal, ROC: annular ring
    ax = axes[2]
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1)
    ax.plot(0.5, 0, 'rx', markersize=12, markeredgewidth=2, label='Poles')
    ax.plot(1.5, 0, 'rx', markersize=12, markeredgewidth=2)
    # ROC: 0.5 < |z| < 1.5
    for r in [0.5, 1.5]:
        ax.plot(r * np.cos(theta), r * np.sin(theta), 'b--', linewidth=1)
    # Shade annular region
    theta_fill = np.linspace(0, 2 * np.pi, 100)
    r_inner, r_outer = 0.5, 1.5
    ax.fill_between(
        np.concatenate([r_inner * np.cos(theta_fill),
                        r_outer * np.cos(theta_fill[::-1])]),
        np.concatenate([r_inner * np.sin(theta_fill),
                        r_outer * np.sin(theta_fill[::-1])]),
        alpha=0.3, color='yellow', label='ROC: 0.5 < |z| < 1.5'
    )
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.set_title('Two-sided signal\nROC: 0.5 < |z| < 1.5')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('roc_visualization.png', dpi=150)
    plt.show()

visualize_roc()
```

---

## 3. Z 변환의 성질(Properties of the Z-Transform)

### 3.1 선형성(Linearity)

$$a \, x_1[n] + b \, x_2[n] \quad \xleftrightarrow{\mathcal{Z}} \quad a \, X_1(z) + b \, X_2(z)$$

ROC: 적어도 $\text{ROC}_1 \cap \text{ROC}_2$ (극점-영점 상쇄가 발생하면 더 클 수 있음).

### 3.2 시간 이동(Time Shifting)

$$x[n - n_0] \quad \xleftrightarrow{\mathcal{Z}} \quad z^{-n_0} X(z)$$

ROC: $X(z)$와 동일 ($z = 0$ 또는 $z = \infty$ 추가/제거 가능).

> 지연 연산자 $z^{-1}$은 디지털 시스템의 기본 요소입니다. 한 샘플의 지연은 $z^{-1}$ 곱셈으로 표현됩니다.

### 3.3 z 영역에서의 스케일링(Scaling in the z-Domain)

$$a^n x[n] \quad \xleftrightarrow{\mathcal{Z}} \quad X(z/a)$$

ROC: $|a| \cdot R^{-} < |z| < |a| \cdot R^{+}$ (ROC가 $|a|$만큼 스케일링됨).

### 3.4 시간 역전(Time Reversal)

$$x[-n] \quad \xleftrightarrow{\mathcal{Z}} \quad X(z^{-1})$$

ROC: $1/R^{+} < |z| < 1/R^{-}$ (ROC가 역전됨).

### 3.5 z 영역에서의 미분(Differentiation in z-Domain)

$$n \, x[n] \quad \xleftrightarrow{\mathcal{Z}} \quad -z \frac{dX(z)}{dz}$$

$n \cdot a^n$ 관련 변환 도출에 유용합니다.

### 3.6 컨볼루션(Convolution)

$$x_1[n] * x_2[n] \quad \xleftrightarrow{\mathcal{Z}} \quad X_1(z) \cdot X_2(z)$$

ROC: 적어도 $\text{ROC}_1 \cap \text{ROC}_2$.

시스템 분석에서 가장 중요한 성질입니다: 시간 영역의 컨볼루션이 z 영역에서 곱셈으로 변환됩니다.

### 3.7 초기값 정리(Initial Value Theorem, 인과 신호)

$$x[0] = \lim_{z \to \infty} X(z)$$

### 3.8 최종값 정리(Final Value Theorem)

$(1 - z^{-1})X(z)$의 모든 극점이 단위원 내부에 있는 경우:

$$\lim_{n \to \infty} x[n] = \lim_{z \to 1} (1 - z^{-1}) X(z)$$

### 3.9 성질 요약표

| 성질 | 시간 영역 | Z 영역 | ROC |
|----------|------------|----------|-----|
| 선형성 | $ax_1 + bx_2$ | $aX_1 + bX_2$ | $\supseteq R_1 \cap R_2$ |
| 시간 이동 | $x[n-n_0]$ | $z^{-n_0}X(z)$ | $R$ (경우에 따라 $\pm$ 0, $\infty$) |
| 스케일링 | $a^n x[n]$ | $X(z/a)$ | $|a| \cdot R$ |
| 역전 | $x[-n]$ | $X(1/z)$ | $1/R$ |
| 미분 | $nx[n]$ | $-z\frac{dX}{dz}$ | $R$ |
| 컨볼루션 | $x_1 * x_2$ | $X_1 X_2$ | $\supseteq R_1 \cap R_2$ |
| 누적 | $\sum_{k=-\infty}^{n} x[k]$ | $\frac{X(z)}{1-z^{-1}}$ | $R \cap \{|z|>1\}$ |

```python
def demonstrate_z_properties():
    """Numerically verify Z-transform properties."""
    # Test signal: x[n] = 0.8^n * u[n], truncated to 100 samples
    N = 100
    a = 0.8
    n = np.arange(N)
    x = a ** n

    # Evaluate Z-transforms on the unit circle
    omega = np.linspace(-np.pi, np.pi, 512)
    z = np.exp(1j * omega)

    def zt_on_circle(signal, omega_vals):
        """Compute Z-transform on unit circle (= DTFT)."""
        z_vals = np.exp(1j * omega_vals)
        result = np.zeros(len(omega_vals), dtype=complex)
        for k in range(len(signal)):
            result += signal[k] * z_vals ** (-k)
        return result

    # Property 1: Time shift
    m = 5
    x_shifted = np.zeros(N + m)
    x_shifted[m:m + N] = x

    X_orig = zt_on_circle(x, omega)
    X_shifted_direct = zt_on_circle(x_shifted, omega)
    X_shifted_property = np.exp(-1j * omega * m) * X_orig

    # Property 2: Convolution
    h = 0.5 ** n  # Another causal exponential
    y_conv = np.convolve(x[:50], h[:50])  # Linear convolution

    X_x = zt_on_circle(x[:50], omega)
    H_z = zt_on_circle(h[:50], omega)
    Y_product = X_x * H_z
    Y_conv_direct = zt_on_circle(y_conv, omega)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Time shift verification
    axes[0, 0].plot(omega / np.pi, np.abs(X_shifted_direct), 'b-',
                    linewidth=2, label='Direct')
    axes[0, 0].plot(omega / np.pi, np.abs(X_shifted_property), 'r--',
                    linewidth=1, label='z^{-m} X(z)')
    axes[0, 0].set_title(f'Time Shift Property (m={m}): Magnitude')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(omega / np.pi,
                    np.abs(X_shifted_direct - X_shifted_property), 'k-')
    axes[0, 1].set_title(f'Time Shift Error')
    axes[0, 1].set_ylabel('|Error|')
    axes[0, 1].grid(True, alpha=0.3)

    # Convolution verification
    axes[1, 0].plot(omega / np.pi, np.abs(Y_conv_direct), 'b-',
                    linewidth=2, label='Z{x*h}')
    axes[1, 0].plot(omega / np.pi, np.abs(Y_product), 'r--',
                    linewidth=1, label='X(z)H(z)')
    axes[1, 0].set_title('Convolution Property: Magnitude')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(omega / np.pi, np.abs(Y_conv_direct - Y_product), 'k-')
    axes[1, 1].set_title('Convolution Property Error')
    axes[1, 1].set_ylabel('|Error|')
    axes[1, 1].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel(r'$\omega / \pi$')

    plt.tight_layout()
    plt.savefig('z_properties.png', dpi=150)
    plt.show()

demonstrate_z_properties()
```

---

## 4. 역 Z 변환(Inverse Z-Transform)

### 4.1 형식적 정의

$$x[n] = \frac{1}{2\pi j} \oint_C X(z) \, z^{n-1} \, dz$$

여기서 $C$는 ROC 내에서 원점을 둘러싸는 반시계 방향 경로입니다.

실제로는 세 가지 방법이 일반적으로 사용됩니다:

### 4.2 방법 1: 부분 분수 전개(Partial Fraction Expansion)

유리 함수 $X(z) = B(z)/A(z)$에 대해, 단순 분수로 분해합니다:

$$X(z) = \sum_i \frac{A_i}{1 - p_i z^{-1}} + \cdots$$

각 항은 알려진 역 Z 변환을 가지며, ROC가 각 항이 인과적인지 반인과적인지를 결정합니다.

**예시:**

$$X(z) = \frac{1}{(1 - 0.5z^{-1})(1 - 0.8z^{-1})}, \quad |z| > 0.8$$

부분 분수 전개:

$$X(z) = \frac{A}{1 - 0.5z^{-1}} + \frac{B}{1 - 0.8z^{-1}}$$

풀면: $A = \frac{-5}{3}$, $B = \frac{8}{3}$

ROC가 $|z| > 0.8$이므로(두 극점 모두 ROC 내부), 두 항 모두 인과적:

$$x[n] = \left(-\frac{5}{3}(0.5)^n + \frac{8}{3}(0.8)^n\right) u[n]$$

```python
def partial_fraction_inverse_z():
    """Inverse Z-transform via partial fraction expansion."""
    # X(z) = 1 / ((1 - 0.5 z^{-1})(1 - 0.8 z^{-1}))
    # Numerator: [1] (in z^{-1} form)
    # Denominator: (1 - 0.5 z^{-1})(1 - 0.8 z^{-1})
    #            = 1 - 1.3 z^{-1} + 0.4 z^{-2}

    # Using scipy.signal for partial fractions
    # Express as H(z) = B(z)/A(z) in descending powers of z
    # B(z) = 1
    # A(z) = 1 - 1.3 z^{-1} + 0.4 z^{-2}

    b = [1.0]                   # Numerator coefficients
    a = [1.0, -1.3, 0.4]       # Denominator coefficients

    # Partial fraction expansion
    # scipy uses z (not z^{-1}), so we need to be careful
    # Convert to z-form: multiply num/den by z^2
    b_z = [0, 0, 1]            # z^0 (need to match length)
    a_z = [1, -1.3, 0.4]       # z^2 - 1.3z + 0.4

    residues, poles, remainder = signal.residuez(b, a)

    print("Partial Fraction Expansion")
    print("=" * 50)
    print(f"X(z) = 1 / ((1 - 0.5z^-1)(1 - 0.8z^-1))")
    print(f"\nPoles: {poles}")
    print(f"Residues: {residues}")
    print(f"Remainder: {remainder}")

    # Reconstruct x[n]
    N = 30
    n = np.arange(N)

    # From partial fractions (causal, ROC: |z| > 0.8)
    x_pf = np.zeros(N)
    for r, p in zip(residues, poles):
        x_pf += np.real(r * p ** n)

    # Direct computation via scipy.signal (impulse response)
    _, x_impulse = signal.dimpulse(signal.dlti(b, a, dt=1), n=N)
    x_impulse = np.squeeze(x_impulse)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stem(n, x_pf, linefmt='b-', markerfmt='bo', basefmt='k-',
            label='Partial fractions')
    ax.plot(n, x_impulse, 'rx', markersize=8, label='scipy dimpulse')
    ax.set_title('Inverse Z-Transform via Partial Fractions')
    ax.set_xlabel('n')
    ax.set_ylabel('x[n]')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('inverse_z_partial.png', dpi=150)
    plt.show()

partial_fraction_inverse_z()
```

### 4.3 방법 2: 긴 나눗셈(Long Division, 멱급수 전개)

$B(z^{-1})$을 $A(z^{-1})$로 나누어 $z^{-n}$의 계수를 구하면, 이 계수들이 $x[n]$의 값이 됩니다.

**예시:**

$$X(z) = \frac{1}{1 - 1.5z^{-1} + 0.5z^{-2}}$$

$z^{-1}$에서의 긴 나눗셈:

$$\frac{1}{1 - 1.5z^{-1} + 0.5z^{-2}} = 1 + 1.5z^{-1} + 1.75z^{-2} + 1.875z^{-3} + \cdots$$

따라서 $x[0] = 1$, $x[1] = 1.5$, $x[2] = 1.75$, $x[3] = 1.875$, ...

```python
def long_division_inverse_z():
    """Inverse Z-transform via long division."""
    # X(z) = B(z^{-1}) / A(z^{-1})
    b = np.array([1.0])
    a = np.array([1.0, -1.5, 0.5])

    N = 20
    x = np.zeros(N)

    # Long division algorithm
    remainder = np.zeros(len(b) + N)
    remainder[:len(b)] = b

    for n in range(N):
        x[n] = remainder[0] / a[0]
        for k in range(len(a)):
            if k < len(remainder):
                remainder[k] -= x[n] * a[k]
        remainder = np.roll(remainder, -1)
        remainder[-1] = 0

    # Verify with scipy
    _, x_scipy = signal.dimpulse(signal.dlti(b, a, dt=1), n=N)
    x_scipy = np.squeeze(x_scipy)

    print("Long Division Inverse Z-Transform")
    print("=" * 40)
    print(f"X(z) = 1 / (1 - 1.5z^(-1) + 0.5z^(-2))")
    print(f"\n{'n':>4s} | {'x[n] (long div)':>16s} | {'x[n] (scipy)':>14s}")
    print("-" * 40)
    for i in range(min(10, N)):
        print(f"{i:4d} | {x[i]:16.6f} | {x_scipy[i]:14.6f}")

long_division_inverse_z()
```

### 4.4 방법 3: 경로 적분(Contour Integration, 유수 정리)

$$x[n] = \sum_{\text{poles } p_k \text{ inside } C} \text{Res}\left[X(z) z^{n-1}, p_k\right]$$

$z = p_k$에서의 단순 극점에 대해:

$$\text{Res}\left[X(z)z^{n-1}, p_k\right] = \lim_{z \to p_k} (z - p_k) X(z) z^{n-1}$$

---

## 5. 전달 함수 H(z)

### 5.1 정의

차분 방정식으로 기술되는 LTI 시스템에 대해:

$$\sum_{k=0}^{N} a_k \, y[n-k] = \sum_{k=0}^{M} b_k \, x[n-k]$$

전달 함수(transfer function)는:

$$\boxed{H(z) = \frac{Y(z)}{X(z)} = \frac{\sum_{k=0}^{M} b_k z^{-k}}{\sum_{k=0}^{N} a_k z^{-k}} = \frac{B(z)}{A(z)}}$$

z 영역에서의 출력은 단순히:

$$Y(z) = H(z) \cdot X(z)$$

### 5.2 임펄스 응답과 전달 함수

임펄스 응답(impulse response) $h[n]$은 $H(z)$의 역 Z 변환입니다:

$$h[n] = \mathcal{Z}^{-1}\{H(z)\}$$

$Y(z) = H(z) X(z)$이고 z 영역의 곱셈이 시간 영역의 컨볼루션에 대응하므로:

$$y[n] = h[n] * x[n] = \sum_{k=-\infty}^{\infty} h[k] \, x[n-k]$$

### 5.3 예시: 1차 시스템

$$y[n] = 0.9 \, y[n-1] + x[n]$$

전달 함수:

$$H(z) = \frac{1}{1 - 0.9z^{-1}} = \frac{z}{z - 0.9}$$

- $z = 0.9$에서 극점 하나
- $z = 0$에서 영점 하나 ($z^{-1}$ 형식에서의 자명한 영점)

```python
def transfer_function_example():
    """Analyze a first-order digital system."""
    # y[n] = 0.9 * y[n-1] + x[n]
    # H(z) = 1 / (1 - 0.9 z^{-1})
    b = [1.0]
    a = [1.0, -0.9]

    # Create discrete-time system
    sys = signal.dlti(b, a, dt=1)

    # Impulse response
    t_imp, h = signal.dimpulse(sys, n=40)
    h = np.squeeze(h)

    # Step response
    t_step, s = signal.dstep(sys, n=40)
    s = np.squeeze(s)

    # Frequency response
    w, H = signal.freqz(b, a, worN=1024)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Impulse response
    axes[0, 0].stem(np.arange(len(h)), h, linefmt='b-', markerfmt='bo',
                    basefmt='k-')
    axes[0, 0].set_title('Impulse Response h[n]')
    axes[0, 0].set_xlabel('n')
    axes[0, 0].set_ylabel('h[n]')
    axes[0, 0].grid(True, alpha=0.3)

    # Step response
    axes[0, 1].stem(np.arange(len(s)), s, linefmt='r-', markerfmt='ro',
                    basefmt='k-')
    axes[0, 1].set_title('Step Response')
    axes[0, 1].set_xlabel('n')
    axes[0, 1].set_ylabel('y[n]')
    axes[0, 1].grid(True, alpha=0.3)

    # Magnitude response
    axes[1, 0].plot(w / np.pi, 20 * np.log10(np.abs(H)), 'b-', linewidth=2)
    axes[1, 0].set_title('Magnitude Response |H(e^jw)|')
    axes[1, 0].set_xlabel(r'$\omega / \pi$')
    axes[1, 0].set_ylabel('Magnitude (dB)')
    axes[1, 0].grid(True, alpha=0.3)

    # Phase response
    axes[1, 1].plot(w / np.pi, np.unwrap(np.angle(H)), 'r-', linewidth=2)
    axes[1, 1].set_title('Phase Response')
    axes[1, 1].set_xlabel(r'$\omega / \pi$')
    axes[1, 1].set_ylabel('Phase (radians)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(r'System: $y[n] = 0.9\,y[n-1] + x[n]$', fontsize=14)
    plt.tight_layout()
    plt.savefig('transfer_function.png', dpi=150)
    plt.show()

transfer_function_example()
```

---

## 6. z 평면의 극점과 영점(Poles and Zeros in the z-Plane)

### 6.1 정의

유리(rational) 전달 함수에 대해:

$$H(z) = \frac{b_0 + b_1 z^{-1} + \cdots + b_M z^{-M}}{a_0 + a_1 z^{-1} + \cdots + a_N z^{-N}} = G \cdot \frac{\prod_{k=1}^{M}(z - z_k)}{\prod_{k=1}^{N}(z - p_k)}$$

- **영점(zeros)** ($z_k$): $H(z) = 0$이 되는 $z$ 값 (분자의 근)
- **극점(poles)** ($p_k$): $H(z) \to \infty$가 되는 $z$ 값 (분모의 근)
- $G$: 이득 인자(gain factor)

### 6.2 극점-영점 도식(Pole-Zero Plot)

```python
def pole_zero_analysis():
    """Analyze a system using pole-zero plots."""
    # Second-order system (resonator)
    # H(z) = 1 / (1 - 2r cos(w0) z^{-1} + r^2 z^{-2})
    r = 0.9       # Pole radius (< 1 for stability)
    w0 = np.pi / 4  # Resonant frequency (pi/4 = fs/8)

    b = [1.0]
    a = [1.0, -2 * r * np.cos(w0), r ** 2]

    # Find poles and zeros
    zeros = np.roots(b)
    poles = np.roots(a)

    print("Pole-Zero Analysis")
    print("=" * 50)
    print(f"Zeros: {zeros}")
    print(f"Poles: {poles}")
    print(f"Pole magnitudes: {np.abs(poles)}")
    print(f"Pole angles: {np.angle(poles) / np.pi} * pi")
    print(f"Stable: {all(np.abs(poles) < 1)}")

    # Pole-zero plot and frequency response
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Pole-zero plot
    ax = axes[0]
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1,
            label='Unit circle')

    # Plot zeros
    if len(zeros) > 0:
        ax.plot(np.real(zeros), np.imag(zeros), 'bo', markersize=10,
                label=f'Zeros ({len(zeros)})')

    # Plot poles
    ax.plot(np.real(poles), np.imag(poles), 'rx', markersize=12,
            markeredgewidth=2, label=f'Poles ({len(poles)})')

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title('Pole-Zero Plot')
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Frequency response
    ax = axes[1]
    w, H = signal.freqz(b, a, worN=1024)
    ax.plot(w / np.pi, 20 * np.log10(np.abs(H)), 'b-', linewidth=2)
    ax.axvline(w0 / np.pi, color='red', linestyle='--', alpha=0.5,
               label=f'Resonant freq = {w0/np.pi:.2f}pi')
    ax.set_title('Magnitude Response')
    ax.set_xlabel(r'$\omega / \pi$')
    ax.set_ylabel('Magnitude (dB)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pole_zero.png', dpi=150)
    plt.show()

pole_zero_analysis()
```

### 6.3 극점과 영점 위치의 효과

| 위치 | 효과 |
|----------|--------|
| **단위원 근처의 극점** | 극점 각도에서의 주파수 응답에 날카로운 피크 |
| **단위원 위의 영점** | 영점 각도에서 영(null) 이득 |
| **단위원 내부의 극점** | 안정적, 감소하는 임펄스 응답 |
| **단위원 외부의 극점** | 불안정, 증가하는 임펄스 응답 |
| **단위원 위의 극점** | 경계 안정, 지속적 진동 |
| **원점의 극점** | 순수 지연 (FIR 동작) |
| **켤레 복소 극점** | 공진 (진동적 감소) |

### 6.4 극점-영점 상호작용 탐색

```python
def pole_zero_effects():
    """Show how pole/zero locations affect frequency response."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))

    configurations = [
        {
            'title': 'Lowpass (pole near z=1)',
            'b': [1.0], 'a': [1.0, -0.9]
        },
        {
            'title': 'Highpass (pole near z=-1)',
            'b': [1.0, -1.0], 'a': [1.0, -0.9]
        },
        {
            'title': 'Bandpass (complex conjugate poles)',
            'b': [1.0, 0, -1.0],
            'a': [1.0, -2 * 0.9 * np.cos(np.pi / 4), 0.81]
        },
        {
            'title': 'Notch (zeros on unit circle)',
            'b': [1.0, -2 * np.cos(np.pi / 4), 1.0],
            'a': [1.0, -2 * 0.9 * np.cos(np.pi / 4), 0.81]
        },
        {
            'title': 'All-pass (poles/zeros reciprocal)',
            'b': [0.5, 1.0],
            'a': [1.0, 0.5]
        },
        {
            'title': 'Comb filter (pole at z^N=r^N)',
            'b': [1.0],
            'a': np.concatenate([[1.0], np.zeros(7), [-0.8]])
        },
    ]

    for ax_row, config in zip(axes.flat, configurations):
        b, a = np.array(config['b']), np.array(config['a'])

        w, H = signal.freqz(b, a, worN=1024)
        ax_row.plot(w / np.pi, 20 * np.log10(np.abs(H) + 1e-12),
                    'b-', linewidth=2)
        ax_row.set_title(config['title'])
        ax_row.set_xlabel(r'$\omega / \pi$')
        ax_row.set_ylabel('Magnitude (dB)')
        ax_row.set_ylim(-40, 30)
        ax_row.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pole_zero_effects.png', dpi=150)
    plt.show()

pole_zero_effects()
```

---

## 7. 안정성 분석(Stability Analysis)

### 7.1 BIBO 안정성(BIBO Stability)

인과적 LTI 시스템이 **유계 입력-유계 출력(BIBO, Bounded-Input Bounded-Output) 안정**인 필요충분조건:

$$\sum_{n=0}^{\infty} |h[n]| < \infty$$

Z 변환 관점에서, 인과적 시스템이 BIBO 안정인 필요충분조건:

$$\boxed{\text{$H(z)$의 모든 극점이 단위원 엄격히 내부에 위치: } |p_k| < 1 \, \forall k}$$

### 7.2 안정성 조건

| 극점 위치 | 안정성 | 임펄스 응답 |
|--------------|-----------|-----------------|
| 모든 $|p_k| < 1$ | 안정 | 영으로 수렴 |
| 일부 $|p_k| = 1$ (단순) | 경계 안정 | 유계, 비소멸 |
| 임의 $|p_k| > 1$ | 불안정 | 무한 증가 |
| $|p_k| = 1$ (중복) | 불안정 | $n^{m-1}$처럼 증가 |

### 7.3 안정성 검사

```python
def stability_analysis():
    """Analyze stability of several systems."""
    systems = [
        {
            'name': 'Stable: y[n] = 0.5*y[n-1] + x[n]',
            'b': [1.0], 'a': [1.0, -0.5]
        },
        {
            'name': 'Marginally stable: y[n] = y[n-1] + x[n]',
            'b': [1.0], 'a': [1.0, -1.0]
        },
        {
            'name': 'Unstable: y[n] = 1.1*y[n-1] + x[n]',
            'b': [1.0], 'a': [1.0, -1.1]
        },
        {
            'name': 'Stable oscillator: r=0.9, w0=pi/4',
            'b': [1.0],
            'a': [1.0, -2 * 0.9 * np.cos(np.pi / 4), 0.81]
        },
        {
            'name': 'Unstable oscillator: r=1.05, w0=pi/4',
            'b': [1.0],
            'a': [1.0, -2 * 1.05 * np.cos(np.pi / 4), 1.05**2]
        },
    ]

    fig, axes = plt.subplots(len(systems), 2, figsize=(14, 3 * len(systems)))

    for i, sys_info in enumerate(systems):
        b, a = np.array(sys_info['b']), np.array(sys_info['a'])
        poles = np.roots(a)
        max_pole_mag = np.max(np.abs(poles))

        stability = "STABLE" if max_pole_mag < 1 else \
                    "MARGINALLY STABLE" if np.isclose(max_pole_mag, 1) else \
                    "UNSTABLE"

        # Pole-zero plot
        ax_pz = axes[i, 0]
        theta = np.linspace(0, 2 * np.pi, 200)
        ax_pz.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=0.5)
        ax_pz.plot(np.real(poles), np.imag(poles), 'rx', markersize=10,
                   markeredgewidth=2)
        ax_pz.set_xlim(-1.5, 1.5)
        ax_pz.set_ylim(-1.5, 1.5)
        ax_pz.set_aspect('equal')
        ax_pz.set_title(f'{sys_info["name"]}\n[{stability}] max|pole|={max_pole_mag:.3f}')
        ax_pz.axhline(0, color='gray', linewidth=0.5)
        ax_pz.axvline(0, color='gray', linewidth=0.5)
        ax_pz.grid(True, alpha=0.3)

        # Impulse response
        ax_ir = axes[i, 1]
        N = 40
        h = np.zeros(N)
        h[0] = b[0] / a[0]
        for n in range(1, N):
            h[n] = (b[n] if n < len(b) else 0)
            for k in range(1, min(n + 1, len(a))):
                h[n] -= a[k] * h[n - k]
            h[n] /= a[0]

        color = 'green' if stability == "STABLE" else \
                'orange' if stability == "MARGINALLY STABLE" else 'red'
        ax_ir.stem(np.arange(N), h, linefmt=f'{color[0]}-',
                   markerfmt=f'{color[0]}o', basefmt='k-')
        ax_ir.set_title(f'Impulse Response')
        ax_ir.set_xlabel('n')
        ax_ir.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('stability.png', dpi=150)
    plt.show()

stability_analysis()
```

### 7.4 주리 안정성 검정(Jury Stability Test)

다항식 $A(z) = a_0 z^N + a_1 z^{N-1} + \cdots + a_N$에 대해, 주리 안정성 검정은 명시적으로 근을 계산하지 않고도 모든 근이 단위원 내부에 있는지 필요충분조건을 제공합니다.

**필요 조건 (빠른 확인):**
1. $A(1) > 0$
2. $(-1)^N A(-1) > 0$
3. $|a_N| < |a_0|$

```python
def jury_test(a):
    """Perform Jury stability test on polynomial coefficients."""
    a = np.array(a, dtype=float)
    N = len(a) - 1  # Polynomial order

    print("Jury Stability Test")
    print("=" * 50)
    print(f"Polynomial order: {N}")
    print(f"Coefficients: {a}")

    # Necessary conditions
    A_1 = np.polyval(a, 1)
    A_neg1 = np.polyval(a, -1)
    cond1 = A_1 > 0
    cond2 = ((-1) ** N * A_neg1) > 0
    cond3 = abs(a[-1]) < abs(a[0])

    print(f"\nNecessary conditions:")
    print(f"  A(1) = {A_1:.4f} > 0 ? {cond1}")
    print(f"  (-1)^N * A(-1) = {(-1)**N * A_neg1:.4f} > 0 ? {cond2}")
    print(f"  |a_N| = {abs(a[-1]):.4f} < |a_0| = {abs(a[0]):.4f} ? {cond3}")

    if not (cond1 and cond2 and cond3):
        print("\n  => UNSTABLE (necessary condition violated)")
        return False

    # Verify with actual roots
    roots = np.roots(a)
    max_mag = np.max(np.abs(roots))
    stable = max_mag < 1
    print(f"\nVerification: max|root| = {max_mag:.6f}")
    print(f"System is {'STABLE' if stable else 'UNSTABLE'}")
    return stable

# Test examples
print("System 1:")
jury_test([1, -1.3, 0.4])  # Stable (poles at 0.5, 0.8)
print("\nSystem 2:")
jury_test([1, -2.0, 1.1])  # Unstable
```

---

## 8. DTFT 및 라플라스 변환과의 관계

### 8.1 Z 변환과 DTFT

DTFT는 단위원 위에서 평가한 Z 변환입니다:

$$\boxed{X(e^{j\omega}) = X(z)\big|_{z=e^{j\omega}}}$$

이 관계는 $X(z)$의 ROC가 단위원을 포함하는 경우에 성립합니다.

**결과:** 이산 시간 시스템의 주파수 응답은:

$$H(e^{j\omega}) = H(z)\big|_{z=e^{j\omega}}$$

### 8.2 Z 변환과 라플라스 변환

샘플링된 신호 $x_s(t) = \sum_n x(nT_s) \delta(t - nT_s)$에 대해, 라플라스 변환 $X_s(s)$와 Z 변환 $X(z)$ 사이의 관계는:

$$\boxed{z = e^{sT_s}}$$

또는 동등하게:

$$s = \frac{1}{T_s} \ln z$$

이는 다음과 같이 대응합니다:
- s 평면의 좌반면 ($\text{Re}(s) < 0$) $\to$ 단위원 내부 ($|z| < 1$)
- 허수축 ($\text{Re}(s) = 0$) $\to$ 단위원 ($|z| = 1$)
- s 평면의 우반면 ($\text{Re}(s) > 0$) $\to$ 단위원 외부 ($|z| > 1$)

### 8.3 H(z)로부터의 주파수 응답

특정 물리적 주파수 $f$ (Hz)에서의 주파수 응답 계산:

$$\omega = 2\pi f / f_s \quad \text{(정규화된 디지털 주파수)}$$
$$H(e^{j\omega}) = H(z)\big|_{z = e^{j2\pi f/f_s}}$$

```python
def frequency_response_from_hz():
    """Compute frequency response from H(z) by evaluating on unit circle."""
    # System: H(z) = (1 + z^{-1}) / (1 - 0.5 z^{-1})
    b = [1.0, 1.0]
    a = [1.0, -0.5]

    fs = 8000  # Hz

    # Method 1: Using scipy.signal.freqz
    w, H_scipy = signal.freqz(b, a, worN=1024)
    f_hz = w * fs / (2 * np.pi)

    # Method 2: Direct evaluation on unit circle
    omega = np.linspace(0, np.pi, 1024)
    z = np.exp(1j * omega)
    H_direct = (b[0] + b[1] * z**(-1)) / (a[0] + a[1] * z**(-1))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(f_hz, 20 * np.log10(np.abs(H_scipy)), 'b-',
                 linewidth=2, label='scipy.signal.freqz')
    axes[0].plot(f_hz, 20 * np.log10(np.abs(H_direct)), 'r--',
                 linewidth=1, label='Direct evaluation')
    axes[0].set_title(r'Frequency Response: $H(z) = (1+z^{-1})/(1-0.5z^{-1})$')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(f_hz, np.unwrap(np.angle(H_scipy)) * 180 / np.pi, 'b-',
                 linewidth=2, label='scipy.signal.freqz')
    axes[1].plot(f_hz, np.unwrap(np.angle(H_direct)) * 180 / np.pi, 'r--',
                 linewidth=1, label='Direct evaluation')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Phase (degrees)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('freq_response_hz.png', dpi=150)
    plt.show()

frequency_response_from_hz()
```

### 8.4 s 평면에서 z 평면으로의 사상

```python
def s_to_z_mapping():
    """Visualize the mapping from s-plane to z-plane."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    Ts = 1.0  # Normalized sampling period

    # s-plane
    ax = axes[0]
    # Stability boundary (imaginary axis)
    sigma = np.linspace(-3, 3, 100)
    omega_s = np.linspace(-np.pi / Ts, np.pi / Ts, 100)

    ax.axvline(0, color='red', linewidth=2, label='Stability boundary')
    ax.fill_betweenx([-4, 4], -4, 0, alpha=0.1, color='green',
                      label='Stable region')
    ax.fill_betweenx([-4, 4], 0, 4, alpha=0.1, color='red',
                      label='Unstable region')

    # Constant sigma lines
    for sig in [-2, -1, -0.5, 0.5, 1, 2]:
        ax.axvline(sig, color='gray', linestyle=':', alpha=0.3)

    # Constant omega lines
    for om in np.linspace(-3, 3, 7):
        ax.axhline(om, color='gray', linestyle=':', alpha=0.3)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-4, 4)
    ax.set_title('s-Plane')
    ax.set_xlabel(r'$\sigma$ (Real)')
    ax.set_ylabel(r'$j\omega$ (Imaginary)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # z-plane
    ax = axes[1]
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'r-', linewidth=2,
            label='Unit circle (stability)')

    # Map constant sigma lines
    for sig in [-2, -1, -0.5, 0, 0.5, 1, 2]:
        r = np.exp(sig * Ts)
        ax.plot(r * np.cos(theta), r * np.sin(theta), '--',
                alpha=0.4, label=f'sigma={sig}' if sig in [-1, 0, 1] else '')

    ax.fill_between(np.cos(theta), np.sin(theta), alpha=0.1, color='green')

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_title(r'z-Plane ($z = e^{sT_s}$)')
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('s_to_z_mapping.png', dpi=150)
    plt.show()

s_to_z_mapping()
```

---

## 9. 주파수 응답의 기하학적 해석

### 9.1 벡터 곱으로서의 주파수 응답

주파수 $\omega$에서의 주파수 응답은 기하학적으로 계산할 수 있습니다:

$$|H(e^{j\omega})| = |G| \cdot \frac{\prod_{k=1}^{M} |e^{j\omega} - z_k|}{\prod_{k=1}^{N} |e^{j\omega} - p_k|}$$

$$\angle H(e^{j\omega}) = \angle G + \sum_{k=1}^{M} \angle(e^{j\omega} - z_k) - \sum_{k=1}^{N} \angle(e^{j\omega} - p_k)$$

$\omega$가 $0$에서 $\pi$까지 변화할 때, 점 $e^{j\omega}$는 단위원의 상반부를 따라 움직입니다. 크기는 영점까지의 거리의 곱을 극점까지의 거리의 곱으로 나눈 값입니다.

```python
def geometric_frequency_response():
    """Visualize the geometric interpretation of frequency response."""
    # System with poles and zeros
    zeros = [0.5 + 0.5j, 0.5 - 0.5j]
    poles = [0.8 * np.exp(1j * np.pi / 3), 0.8 * np.exp(-1j * np.pi / 3)]

    # Build transfer function from poles and zeros
    b = np.real(np.poly(zeros))
    a = np.real(np.poly(poles))

    # Animate for a specific frequency
    omega_target = np.pi / 3  # Target frequency
    z_point = np.exp(1j * omega_target)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Pole-zero plot with vectors
    ax = axes[0]
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=0.5)

    # Zeros
    for z in zeros:
        ax.plot(np.real(z), np.imag(z), 'bo', markersize=10)
        # Vector from zero to point on unit circle
        ax.annotate('', xy=(np.real(z_point), np.imag(z_point)),
                    xytext=(np.real(z), np.imag(z)),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

    # Poles
    for p in poles:
        ax.plot(np.real(p), np.imag(p), 'rx', markersize=12, markeredgewidth=2)
        ax.annotate('', xy=(np.real(z_point), np.imag(z_point)),
                    xytext=(np.real(p), np.imag(p)),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    # Point on unit circle
    ax.plot(np.real(z_point), np.imag(z_point), 'g*', markersize=15,
            label=f'e^(j*{omega_target/np.pi:.2f}*pi)')

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title(f'Geometric Vectors at omega = {omega_target/np.pi:.2f}*pi')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Frequency response
    ax = axes[1]
    w, H = signal.freqz(b, a, worN=1024)
    ax.plot(w / np.pi, 20 * np.log10(np.abs(H) + 1e-12), 'b-', linewidth=2)
    ax.axvline(omega_target / np.pi, color='green', linestyle='--',
               linewidth=2, label=f'omega = {omega_target/np.pi:.2f}*pi')

    H_at_target = np.polyval(b, z_point) / np.polyval(a, z_point)
    ax.plot(omega_target / np.pi, 20 * np.log10(np.abs(H_at_target)),
            'g*', markersize=15)

    ax.set_title('Magnitude Response')
    ax.set_xlabel(r'$\omega / \pi$')
    ax.set_ylabel('Magnitude (dB)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('geometric_freq_response.png', dpi=150)
    plt.show()

geometric_frequency_response()
```

---

## 10. 종합 예시: 시스템 분석 파이프라인

```python
def complete_system_analysis():
    """Complete analysis of a digital system from difference equation to
    frequency response."""
    print("Complete System Analysis")
    print("=" * 60)

    # Difference equation:
    # y[n] - 1.2y[n-1] + 0.72y[n-2] = x[n] - 0.5x[n-1]
    b = [1.0, -0.5]
    a = [1.0, -1.2, 0.72]

    # 1. Transfer function
    print("\n1. Transfer function:")
    print(f"   H(z) = ({b[0]} + {b[1]}z^-1) / "
          f"({a[0]} + {a[1]}z^-1 + {a[2]}z^-2)")

    # 2. Poles and zeros
    zeros = np.roots(b)
    poles = np.roots(a)
    print(f"\n2. Zeros: {zeros}")
    print(f"   Poles: {poles}")
    print(f"   Pole magnitudes: {np.abs(poles)}")
    print(f"   Pole angles: {np.angle(poles) * 180 / np.pi} degrees")

    # 3. Stability
    stable = all(np.abs(poles) < 1)
    print(f"\n3. Stability: {'STABLE' if stable else 'UNSTABLE'}")
    print(f"   Max |pole| = {np.max(np.abs(poles)):.4f}")

    # 4. Partial fraction expansion
    r, p, k = signal.residuez(b, a)
    print(f"\n4. Partial fractions:")
    print(f"   Residues: {r}")
    print(f"   Poles: {p}")
    print(f"   Direct term: {k}")

    # 5. Impulse response (first 50 samples)
    _, h = signal.dimpulse(signal.dlti(b, a, dt=1), n=50)
    h = np.squeeze(h)

    # 6. Frequency response
    w, H = signal.freqz(b, a, worN=1024)

    # Plot everything
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2)

    # Pole-zero plot
    ax1 = fig.add_subplot(gs[0, 0])
    theta = np.linspace(0, 2 * np.pi, 200)
    ax1.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=0.5)
    ax1.plot(np.real(zeros), np.imag(zeros), 'bo', markersize=10, label='Zeros')
    ax1.plot(np.real(poles), np.imag(poles), 'rx', markersize=12,
             markeredgewidth=2, label='Poles')
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.set_title('Pole-Zero Plot')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Impulse response
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.stem(np.arange(len(h)), h, linefmt='b-', markerfmt='bo', basefmt='k-')
    ax2.set_title('Impulse Response h[n]')
    ax2.set_xlabel('n')
    ax2.grid(True, alpha=0.3)

    # Magnitude response
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(w / np.pi, 20 * np.log10(np.abs(H) + 1e-12), 'b-', linewidth=2)
    ax3.set_title('Magnitude Response')
    ax3.set_xlabel(r'$\omega / \pi$')
    ax3.set_ylabel('dB')
    ax3.grid(True, alpha=0.3)

    # Phase response
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(w / np.pi, np.unwrap(np.angle(H)) * 180 / np.pi, 'r-',
             linewidth=2)
    ax4.set_title('Phase Response')
    ax4.set_xlabel(r'$\omega / \pi$')
    ax4.set_ylabel('Degrees')
    ax4.grid(True, alpha=0.3)

    # Group delay
    ax5 = fig.add_subplot(gs[2, 0])
    _, gd = signal.group_delay((b, a), w=1024)
    ax5.plot(w / np.pi, gd, 'g-', linewidth=2)
    ax5.set_title('Group Delay')
    ax5.set_xlabel(r'$\omega / \pi$')
    ax5.set_ylabel('Samples')
    ax5.grid(True, alpha=0.3)

    # Step response
    ax6 = fig.add_subplot(gs[2, 1])
    _, s = signal.dstep(signal.dlti(b, a, dt=1), n=50)
    s = np.squeeze(s)
    ax6.stem(np.arange(len(s)), s, linefmt='m-', markerfmt='mo', basefmt='k-')
    ax6.set_title('Step Response')
    ax6.set_xlabel('n')
    ax6.grid(True, alpha=0.3)

    plt.suptitle(r'System: $y[n] - 1.2y[n-1] + 0.72y[n-2] = x[n] - 0.5x[n-1]$',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig('complete_analysis.png', dpi=150)
    plt.show()

complete_system_analysis()
```

---

## 11. 요약

```
┌─────────────────────────────────────────────────────────────────┐
│                      Z 변환(Z-Transform)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  정의:                                                           │
│    X(z) = Σ x[n] z^{-n}     z = r * e^{jω}                    │
│                                                                  │
│  ROC (수렴 영역):                                                │
│    - 신호를 유일하게 결정 (같은 X(z), 다른 ROC →               │
│      다른 x[n])                                                  │
│    - 인과적: |z| > R⁻  (원 외부)                               │
│    - 반인과적: |z| < R⁺ (원 내부)                              │
│    - 양방향: R⁻ < |z| < R⁺ (환형 고리)                        │
│                                                                  │
│  주요 성질:                                                      │
│    - 시간 이동: x[n-m] ↔ z^{-m} X(z)                          │
│    - 컨볼루션: x₁*x₂ ↔ X₁(z)·X₂(z)                           │
│    - z^{-1} = 한 샘플 지연                                      │
│                                                                  │
│  전달 함수:                                                      │
│    H(z) = Y(z)/X(z) = B(z)/A(z)                                │
│    극점 → 분모의 근                                              │
│    영점 → 분자의 근                                              │
│                                                                  │
│  안정성 (인과 시스템):                                           │
│    모든 극점이 단위원 내부 ↔ BIBO 안정                          │
│                                                                  │
│  관계:                                                           │
│    단위원 위의 Z 변환 = DTFT: X(e^{jω}) = X(z)|_{z=e^{jω}}    │
│    z = e^{sTs}로 Z 변환과 라플라스 변환 연결                   │
│                                                                  │
│  역변환 방법:                                                    │
│    1. 부분 분수 → 알려진 쌍                                     │
│    2. 긴 나눗셈 → 멱급수                                        │
│    3. 경로 적분 → 유수 정리                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 12. 연습 문제

### 연습 문제 1: Z 변환 계산

각각에 대해 Z 변환을 계산하고 ROC를 명시하시오:

**(a)** $x[n] = (0.5)^n u[n] + (0.8)^n u[n]$

**(b)** $x[n] = (0.6)^n u[n] - 2(0.6)^n u[n-3]$

**(c)** $x[n] = n(0.9)^n u[n]$

**(d)** $x[n] = (0.7)^{|n|}$ (양방향)

### 연습 문제 2: 역 Z 변환

부분 분수를 사용하여 각각에 대한 $x[n]$을 구하시오:

**(a)** $X(z) = \frac{z}{(z-0.5)(z-0.8)}, \quad |z| > 0.8$

**(b)** $X(z) = \frac{z}{(z-0.5)(z-0.8)}, \quad |z| < 0.5$

**(c)** $X(z) = \frac{z^2}{(z-0.5)(z-0.8)}, \quad |z| > 0.8$

**(d)** $X(z) = \frac{1+2z^{-1}}{1-z^{-1}+0.5z^{-2}}, \quad |z| > 0.707$

### 연습 문제 3: 시스템 분석

다음 차분 방정식이 주어졌을 때: $y[n] = 0.8y[n-1] - 0.64y[n-2] + x[n] + x[n-1]$

**(a)** $H(z)$를 구하고 극점-영점 도식을 스케치하시오.

**(b)** 시스템이 안정적인가? 이유를 설명하시오.

**(c)** 부분 분수를 사용하여 임펄스 응답 $h[n]$을 구하시오.

**(d)** 주파수 응답(크기 및 위상)을 계산하고 도식화하시오.

**(e)** 이 시스템은 저역 통과, 고역 통과, 대역 통과, 대역 저지 필터 중 어느 것인가?

### 연습 문제 4: 안정성 판정

근을 직접 계산하지 않고 각 시스템의 안정성을 결정하시오:

**(a)** $H(z) = \frac{1}{1 - 0.5z^{-1} + 0.06z^{-2}}$

**(b)** $H(z) = \frac{z^2}{z^2 - 1.4z + 0.85}$

**(c)** $H(z) = \frac{1}{1 - 1.8\cos(0.4\pi)z^{-1} + 0.81z^{-2}}$

주리 안정성 검정을 사용하고 Python으로 검증하시오.

### 연습 문제 5: ROC와 신호 결정

Z 변환이 $X(z) = \frac{2z^2 - 1.5z}{z^2 - 0.9z + 0.2}$일 때:

**(a)** 가능한 모든 ROC를 구하시오.

**(b)** 각 ROC에 대해 대응하는 신호 $x[n]$ (인과적, 반인과적, 또는 양방향)을 결정하시오.

**(c)** 어느 ROC에서 DTFT가 존재하는가?

### 연습 문제 6: 전달 함수 설계

다음 사양을 가진 2차 디지털 시스템을 설계하시오:
- 공진 주파수: $\omega_0 = \pi/3$ rad/sample
- 대역폭: $\Delta\omega \approx 0.1$ rad/sample
- 공진에서 단위 이득

**(a)** $z = re^{\pm j\omega_0}$에 극점을 놓고 원하는 대역폭을 위한 $r$을 선택하시오. (힌트: $r$이 1에 가까울 때 대역폭 $\approx 2(1-r)$)

**(b)** 공진에서 단위 이득을 얻기 위한 영점을 놓으시오.

**(c)** Python으로 구현하고 주파수 응답을 도식화하시오.

**(d)** 공진 주파수와 다른 주파수를 포함한 신호로 테스트하시오.

### 연습 문제 7: 디지털 발진기 구현

디지털 발진기는 다음 차분 방정식을 사용하여 만들 수 있습니다:

$$y[n] = 2\cos(\omega_0) y[n-1] - y[n-2]$$

**(a)** $H(z)$와 극점을 구하시오. 극점은 어디에 위치하는가?

**(b)** 왜 이 시스템이 경계 안정인가? 유한 정밀도 산술에서 실제로 어떤 일이 발생하는가?

**(c)** Python으로 발진기를 구현하고 8000 Hz 샘플링 레이트에서 440 Hz 음을 생성하시오. 10000 샘플 이후 직접 사인 계산과 비교하시오.

---

## 13. 더 읽을 거리

- Oppenheim, Schafer. *Discrete-Time Signal Processing*, 3rd ed. Chapters 3, 5.
- Proakis, Manolakis. *Digital Signal Processing*, 4th ed. Chapters 3-4.
- Haykin, Van Veen. *Signals and Systems*, 2nd ed. Chapter 8.
- Roberts, M. J. *Signals and Systems*, Chapter 10.
- Phillips, Parr, Riskin. *Signals, Systems, and Transforms*, 5th ed. Chapters 8-9.

---

**이전**: [06. 이산 푸리에 변환](06_Discrete_Fourier_Transform.md) | **다음**: [08. 디지털 필터 기초](08_Digital_Filter_Fundamentals.md)
