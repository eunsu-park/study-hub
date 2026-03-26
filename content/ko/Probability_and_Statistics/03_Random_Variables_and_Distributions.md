# 확률변수와 분포

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 확률변수(random variable)를 표본공간에서 실수 직선으로의 가측함수로 정의하기
2. 이산 확률변수와 연속 확률변수 구별하기
3. 확률 질량 함수(PMF), 확률 밀도 함수(PDF), 누적 분포 함수(CDF)를 통해 분포 명세하기
4. 유효한 PMF, PDF, CDF가 만족해야 하는 성질 서술 및 검증하기
5. CDF를 사용하여 구간의 확률 계산하기
6. 분위수 함수(inverse CDF)를 정의하고 계산하기
7. Python 표준 라이브러리를 사용하여 확률변수 시뮬레이션하기

---

## 개요

확률변수(random variable)는 추상적인 실험 결과를 숫자로 변환하여, 확률론에서 미적분학과 대수학의 전체 능력을 활용할 수 있게 합니다. 이 레슨에서는 이 개념을 공식화하고, 확률변수의 분포를 기술하는 세 가지 주요 방법인 확률 질량 함수(Probability Mass Function, PMF), 확률 밀도 함수(Probability Density Function, PDF), 누적 분포 함수(Cumulative Distribution Function, CDF)를 소개합니다.

---

## 목차

1. [확률변수의 정의](#1-확률변수의-정의)
2. [이산 확률변수](#2-이산-확률변수)
3. [연속 확률변수](#3-연속-확률변수)
4. [누적 분포 함수](#4-누적-분포-함수)
5. [분위수 함수](#5-분위수-함수)
6. [혼합 분포](#6-혼합-분포)
7. [확률변수의 함수](#7-확률변수의-함수)
8. [Python 예제](#8-python-예제)
9. [핵심 요약](#9-핵심-요약)

---

## 1. 확률변수의 정의

### 공식적 정의

**확률변수**(random variable) $X$는 표본공간 $\Omega$의 각 결과 $\omega$를 실수에 대응시키는 함수입니다:

$$X : \Omega \to \mathbb{R}$$

엄밀하게는, $X$는 **가측**(measurable)이어야 합니다 -- 모든 보렐 집합(Borel set) $B \subseteq \mathbb{R}$에 대해, 역상 $\{\omega \in \Omega : X(\omega) \in B\}$이 시그마 대수 $\mathcal{F}$의 사건이어야 합니다.

### 직관

표본공간은 추상적인 결과("세 번째 트랜지스터가 먼저 고장남" 같은)로 구성될 수 있습니다. 확률변수는 각 결과에 숫자를 부여하여, 추상적 레이블 대신 실수로 작업할 수 있게 합니다.

**예제**: 주사위 두 개를 던집니다. $X$ = 눈의 합이라 하면, 표본공간은 36개의 결과를 갖지만, $X$는 $\{2, 3, \ldots, 12\}$의 값을 취합니다.

### 표기 관례

- 확률변수: 대문자 ($X$, $Y$, $Z$)
- 관측값: 소문자 ($x$, $y$, $z$)
- $P(X = x)$ 또는 $P(X \leq x)$는 사건 $\{\omega : X(\omega) = x\}$ 또는 $\{\omega : X(\omega) \leq x\}$의 확률을 의미

---

## 2. 이산 확률변수

### 정의

확률변수 $X$가 유한 또는 가산 무한 집합 $\{x_1, x_2, x_3, \ldots\}$의 값을 취하면 **이산**(discrete)이라 합니다.

### 확률 질량 함수 (PMF)

**PMF** $p_X(x)$는 $X$가 $x$와 같을 확률을 나타냅니다:

$$p_X(x) = P(X = x)$$

### PMF의 성질

유효한 PMF는 다음을 만족해야 합니다:

1. **비음성**: $p_X(x) \geq 0$ (모든 $x$에 대해)
2. **정규화**: $\sum_{\text{all } x} p_X(x) = 1$
3. **집합의 확률**: $P(X \in A) = \sum_{x \in A} p_X(x)$

### 예제: 주사위 두 개의 합

| $x$ | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|-----|---|---|---|---|---|---|---|---|----|----|-----|
| $p_X(x)$ | $\frac{1}{36}$ | $\frac{2}{36}$ | $\frac{3}{36}$ | $\frac{4}{36}$ | $\frac{5}{36}$ | $\frac{6}{36}$ | $\frac{5}{36}$ | $\frac{4}{36}$ | $\frac{3}{36}$ | $\frac{2}{36}$ | $\frac{1}{36}$ |

검증: $1 + 2 + 3 + 4 + 5 + 6 + 5 + 4 + 3 + 2 + 1 = 36$이므로 $\sum p_X(x) = 36/36 = 1$.

---

## 3. 연속 확률변수

### 정의

확률변수 $X$가 **연속**(continuous)이란, 임의의 구간 $[a, b]$에 대해 다음을 만족하는 비음 함수 $f_X(x)$가 존재하는 것을 의미합니다:

$$P(a \leq X \leq b) = \int_a^b f_X(x) \, dx$$

### 확률 밀도 함수 (PDF)

함수 $f_X(x)$를 **확률 밀도 함수**(probability density function)라 합니다.

### PDF의 성질

1. **비음성**: $f_X(x) \geq 0$ (모든 $x$에 대해)
2. **정규화**: $\int_{-\infty}^{\infty} f_X(x) \, dx = 1$
3. **구간의 확률**: $P(a \leq X \leq b) = \int_a^b f_X(x) \, dx$

### PMF와의 핵심 차이

연속 확률변수에서 단일 점의 확률은 **0**입니다:

$$P(X = x) = \int_x^x f_X(t) \, dt = 0$$

따라서, 연속 $X$에 대해:

$$P(a \leq X \leq b) = P(a < X < b) = P(a \leq X < b) = P(a < X \leq b)$$

**참고**: $f_X(x)$는 **밀도**이지 확률이 아닙니다. 1을 초과할 수 있습니다 (예: $[0, 1/3]$에서 $f_X(x) = 3$). 구간에 대한 적분만이 확률을 줍니다.

### 예제: $[0, 1]$ 위의 균등분포

$$f_X(x) = \begin{cases} 1 & \text{if } 0 \leq x \leq 1 \\ 0 & \text{otherwise} \end{cases}$$

$$P(0.2 \leq X \leq 0.7) = \int_{0.2}^{0.7} 1 \, dx = 0.5$$

---

## 4. 누적 분포 함수

### 정의

확률변수 $X$의 **CDF** $F_X(x)$는 모든 $x \in \mathbb{R}$에 대해 다음과 같이 정의됩니다:

$$F_X(x) = P(X \leq x)$$

CDF는 이산, 연속, 혼합 확률변수 모두에 적용되는 **보편적** 기술 방법을 제공합니다.

### CDF의 성질

모든 유효한 CDF는 다음을 만족합니다:

1. **비감소**: $a < b$이면 $F_X(a) \leq F_X(b)$
2. **우연속**: $\lim_{x \to a^+} F_X(x) = F_X(a)$
3. **극한**: $\lim_{x \to -\infty} F_X(x) = 0$, $\lim_{x \to \infty} F_X(x) = 1$
4. **유계**: 모든 $x$에 대해 $0 \leq F_X(x) \leq 1$

### CDF로부터 확률 계산

$$P(a < X \leq b) = F_X(b) - F_X(a)$$

$$P(X > a) = 1 - F_X(a)$$

$$P(X = a) = F_X(a) - \lim_{x \to a^-} F_X(x) \quad (\text{$a$에서의 점프 크기})$$

### 이산 확률변수의 CDF

이산 $X$의 PMF가 $p_X$일 때:

$$F_X(x) = \sum_{x_i \leq x} p_X(x_i)$$

CDF는 지지(support)의 각 값에서 점프가 있는 **계단 함수**입니다.

### 연속 확률변수의 CDF

연속 $X$의 PDF가 $f_X$일 때:

$$F_X(x) = \int_{-\infty}^{x} f_X(t) \, dt$$

CDF는 **연속**(점프 없음)이며, 도함수가 존재하는 곳에서 $f_X(x) = F_X'(x)$입니다.

### 예제: 지수분포

PDF: $f_X(x) = \lambda e^{-\lambda x}$ ($x \geq 0$, $\lambda > 0$).

CDF:

$$F_X(x) = \int_0^x \lambda e^{-\lambda t} \, dt = 1 - e^{-\lambda x}, \quad x \geq 0$$

$$P(1 \leq X \leq 3) = F_X(3) - F_X(1) = (1 - e^{-3\lambda}) - (1 - e^{-\lambda}) = e^{-\lambda} - e^{-3\lambda}$$

---

## 5. 분위수 함수

### 정의

$0 < p < 1$에 대한 **분위수 함수**(quantile function, 역CDF) $F_X^{-1}(p)$는:

$$F_X^{-1}(p) = \inf\{x : F_X(x) \geq p\}$$

순증가하고 연속인 CDF의 경우, 이는 $F_X(x) = p$를 $x$에 대해 푸는 것으로 단순화됩니다.

### 특수 분위수

| 이름 | $p$의 값 |
|------|----------|
| 중앙값 (median) | $p = 0.5$ |
| 제1사분위수 ($Q_1$) | $p = 0.25$ |
| 제3사분위수 ($Q_3$) | $p = 0.75$ |
| $k$번째 백분위수 | $p = k/100$ |

### 예제

$X \sim \text{Exponential}(\lambda)$인 경우:

$$F_X(x) = 1 - e^{-\lambda x} = p \implies x = -\frac{\ln(1 - p)}{\lambda}$$

중앙값: $x_{0.5} = \frac{\ln 2}{\lambda}$

### 역변환 표집법

$U \sim \text{Uniform}(0, 1)$이면, $X = F_X^{-1}(U)$는 CDF가 $F_X$인 확률변수입니다. 이것은 확률변수를 시뮬레이션하기 위한 근본적인 기법입니다.

---

## 6. 혼합 분포

**혼합**(mixed) 확률변수는 CDF가 부분적으로 연속이고 부분적으로 점프를 갖는 경우입니다. PMF만으로도, PDF만으로도 기술할 수 없습니다.

**예제**: 콜센터 대기 시간 $X$:

- 확률 0.3으로 즉시 응답 ($X = 0$)
- 확률 0.7로 대기 시간이 지수분포를 따름

$$F_X(x) = \begin{cases} 0 & x < 0 \\ 0.3 + 0.7(1 - e^{-\lambda x}) & x \geq 0 \end{cases}$$

$x = 0$에서 크기 0.3의 점프 후 연속적 증가가 따릅니다.

---

## 7. 확률변수의 함수

$X$가 확률변수이고 $g : \mathbb{R} \to \mathbb{R}$이 (가측) 함수이면, $Y = g(X)$ 역시 확률변수입니다.

### 이산 경우

$X$가 PMF $p_X$를 갖는 이산 확률변수이면, $Y = g(X)$의 PMF는:

$$p_Y(y) = \sum_{\{x : g(x) = y\}} p_X(x)$$

### 연속 경우 (단조 $g$)

$g$가 순단조이고 미분 가능하며 역함수가 $g^{-1}$이면:

$$f_Y(y) = f_X(g^{-1}(y)) \left| \frac{d}{dy} g^{-1}(y) \right|$$

이것이 **변수 변환**(change of variables) 공식입니다 (레슨 08에서 자세히 다룸).

---

## 8. Python 예제

### 이산 확률변수: 주사위 합의 PMF

```python
from collections import Counter

def dice_sum_pmf():
    """Compute and display the PMF of the sum of two fair dice."""
    outcomes = [(d1, d2) for d1 in range(1, 7) for d2 in range(1, 7)]
    sums = [d1 + d2 for d1, d2 in outcomes]
    counts = Counter(sums)

    print("x   P(X=x)     Fraction")
    print("-" * 30)
    total = len(outcomes)  # 36
    for x in range(2, 13):
        prob = counts[x] / total
        print(f"{x:2d}  {prob:.4f}     {counts[x]}/{total}")

    # Verify normalization
    assert sum(counts.values()) == total

dice_sum_pmf()
```

### CDF 계산

```python
def dice_sum_cdf():
    """Compute the CDF of the sum of two fair dice."""
    from collections import Counter

    outcomes = [(d1, d2) for d1 in range(1, 7) for d2 in range(1, 7)]
    sums = [d1 + d2 for d1, d2 in outcomes]
    counts = Counter(sums)

    total = 36
    cumulative = 0.0
    print("x   F(x) = P(X <= x)")
    print("-" * 25)
    for x in range(2, 13):
        cumulative += counts[x] / total
        print(f"{x:2d}  {cumulative:.4f}")

    # P(5 < X <= 9) = F(9) - F(5)
    f9 = sum(counts[k] for k in range(2, 10)) / total
    f5 = sum(counts[k] for k in range(2, 6)) / total
    print(f"\nP(5 < X <= 9) = F(9) - F(5) = {f9:.4f} - {f5:.4f} = {f9 - f5:.4f}")

dice_sum_cdf()
```

### 역변환 표집법

```python
import random
import math

def inverse_transform_exponential(lam=1.0, n=10000):
    """Generate Exponential(lam) samples via inverse transform."""
    random.seed(42)
    samples = []
    for _ in range(n):
        u = random.random()          # U ~ Uniform(0, 1)
        x = -math.log(1 - u) / lam   # F^{-1}(u) for Exponential
        samples.append(x)

    # Check empirical mean and variance
    mean = sum(samples) / n
    var = sum((x - mean) ** 2 for x in samples) / (n - 1)

    print(f"Exponential(lambda={lam}) via inverse transform:")
    print(f"  Theoretical mean = {1/lam:.4f},  Sample mean = {mean:.4f}")
    print(f"  Theoretical var  = {1/lam**2:.4f},  Sample var  = {var:.4f}")

    # Compute empirical CDF at a few points
    for t in [0.5, 1.0, 2.0]:
        empirical = sum(1 for x in samples if x <= t) / n
        theoretical = 1 - math.exp(-lam * t)
        print(f"  F({t}) = {theoretical:.4f} (theoretical), {empirical:.4f} (empirical)")

inverse_transform_exponential()
```

### 혼합 분포 시뮬레이션

```python
import random
import math

def mixed_distribution_sim(n=100000, lam=2.0):
    """Simulate a mixed distribution: P(X=0)=0.3, else Exp(lam)."""
    random.seed(7)
    samples = []
    for _ in range(n):
        if random.random() < 0.3:
            samples.append(0.0)  # Point mass at 0
        else:
            u = random.random()
            samples.append(-math.log(1 - u) / lam)  # Exponential

    # Empirical P(X = 0)
    p_zero = sum(1 for x in samples if x == 0.0) / n
    mean = sum(samples) / n

    # Theoretical mean: 0.3*0 + 0.7*(1/lam) = 0.7/lam
    print(f"P(X = 0): empirical = {p_zero:.4f}, theoretical = 0.3000")
    print(f"E[X]:     empirical = {mean:.4f}, theoretical = {0.7/lam:.4f}")

mixed_distribution_sim()
```

### PDF 정규화의 수치적 검증

```python
def trapezoidal_integrate(f, a, b, n=10000):
    """Numerical integration using the trapezoidal rule."""
    h = (b - a) / n
    total = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        total += f(a + i * h)
    return total * h

import math

# Verify that the standard normal PDF integrates to 1
def standard_normal_pdf(x):
    return math.exp(-x**2 / 2) / math.sqrt(2 * math.pi)

# Integrate from -10 to 10 (effectively -inf to inf for the normal)
result = trapezoidal_integrate(standard_normal_pdf, -10, 10, n=100000)
print(f"Integral of standard normal PDF from -10 to 10: {result:.8f}")
# Should be very close to 1.0
```

---

## 9. 핵심 요약

1. **확률변수**는 결과를 숫자에 대응시키는 함수 $X: \Omega \to \mathbb{R}$입니다. 확률론의 핵심 대상입니다.

2. **이산 확률변수**는 가산 지지를 가지며 **PMF**로 기술됩니다. $p_X(x) = P(X = x)$입니다.

3. **연속 확률변수**는 비가산 지지를 가지며 **PDF**로 기술됩니다. $P(a \leq X \leq b) = \int_a^b f_X(x)\,dx$이며, 단일 점에서의 확률은 0입니다.

4. **CDF** $F_X(x) = P(X \leq x)$는 보편적입니다 -- 이산, 연속, 혼합 확률변수 모두에 적용됩니다. 비감소, 우연속이며 0과 1 사이에 유계입니다.

5. **분위수 함수**는 CDF를 역변환합니다: $F_X^{-1}(p)$는 분포의 비율 $p$ 이하인 값을 나타냅니다. 이를 통해 **역변환 표집법**이 가능합니다.

6. **혼합 분포**는 점질량과 연속 밀도를 결합합니다. 응용에서 자연스럽게 나타납니다 (예: 0일 수 있는 보험 청구).

---

*이전: [02 - 확률 공리와 법칙](./02_Probability_Axioms_and_Rules.md) | 다음: [04 - 기댓값과 적률](./04_Expectation_and_Moments.md)*
