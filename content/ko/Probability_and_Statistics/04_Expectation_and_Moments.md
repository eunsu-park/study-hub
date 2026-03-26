# 기댓값과 적률

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 이산 및 연속 확률변수의 기댓값 계산하기
2. 기댓값의 선형성을 적용하여 계산 간소화하기
3. 분산을 정의와 간편 공식 모두를 사용하여 계산하기
4. 고차 적률(왜도와 첨도)을 정의하고 해석하기
5. 적률 생성 함수(MGF)로부터 적률 유도하기
6. 마르코프, 체비셰프, 옌센 부등식을 적용하여 확률 한계 구하기
7. Python으로 적률 계산 구현하기

---

## 개요

기댓값(평균)은 확률변수의 분포를 요약하는 가장 중요한 단일 수치입니다 -- "질량의 중심"을 알려줍니다. 분산(variance)은 퍼짐을 측정하고, 고차 적률(higher moments)은 분포의 형태를 파악합니다. 적률 생성 함수(Moment Generating Function, MGF)는 모든 적률을 하나의 함수에 담는 우아한 대수적 도구를 제공합니다. 확률 부등식(probability inequalities)은 전체 분포가 알려지지 않아도 유용한 한계를 유도할 수 있게 합니다.

---

## 목차

1. [기댓값](#1-기댓값)
2. [기댓값의 성질](#2-기댓값의-성질)
3. [분산](#3-분산)
4. [고차 적률: 왜도와 첨도](#4-고차-적률-왜도와-첨도)
5. [적률 생성 함수](#5-적률-생성-함수)
6. [확률 부등식](#6-확률-부등식)
7. [Python 예제](#7-python-예제)
8. [핵심 요약](#8-핵심-요약)

---

## 1. 기댓값

### 이산 경우

$X$가 PMF $p_X(x)$를 갖는 이산 확률변수일 때, **기댓값**(expected value, 또는 **평균**)은:

$$E[X] = \mu_X = \sum_{x} x \, p_X(x)$$

단, $\sum_{x} |x| \, p_X(x) < \infty$ (절대 수렴)이어야 합니다.

### 연속 경우

$X$가 PDF $f_X(x)$를 갖는 연속 확률변수일 때:

$$E[X] = \mu_X = \int_{-\infty}^{\infty} x \, f_X(x) \, dx$$

단, $\int_{-\infty}^{\infty} |x| \, f_X(x) \, dx < \infty$이어야 합니다.

### 무의식 통계학자의 법칙 (LOTUS)

$Y = g(X)$의 분포를 먼저 구하지 않고 $E[g(X)]$를 계산하려면:

**이산**: $E[g(X)] = \sum_{x} g(x) \, p_X(x)$

**연속**: $E[g(X)] = \int_{-\infty}^{\infty} g(x) \, f_X(x) \, dx$

이것은 매우 유용합니다 -- 변환된 변수의 기댓값을 원래 분포로부터 직접 계산할 수 있습니다.

### 예제: 주사위 두 개 합의 기댓값

$X$를 합이라 하면, 레슨 03의 PMF를 사용하여:

$$E[X] = \sum_{x=2}^{12} x \cdot p_X(x) = 2 \cdot \frac{1}{36} + 3 \cdot \frac{2}{36} + \cdots + 12 \cdot \frac{1}{36} = 7$$

또는, $X = D_1 + D_2$이고 각 주사위의 $E[D_i] = 3.5$이므로 선형성에 의해 $E[X] = 3.5 + 3.5 = 7$.

---

## 2. 기댓값의 성질

### 기댓값의 선형성

임의의 확률변수 $X$, $Y$ (종속이어도!) 및 상수 $a, b, c$에 대해:

$$E[aX + bY + c] = aE[X] + bE[Y] + c$$

이는 유한 합으로 확장됩니다:

$$E\left[\sum_{i=1}^{n} a_i X_i\right] = \sum_{i=1}^{n} a_i E[X_i]$$

**선형성은 종속성과 무관하게 성립합니다** -- 이것은 확률론에서 가장 강력한 성질 중 하나입니다.

### 단조성

$X \leq Y$ (모든 결과에서)이면, $E[X] \leq E[Y]$.

### 상수의 기댓값

$$E[c] = c$$

### 비음 확률변수

$X \geq 0$이면, $E[X] \geq 0$.

### 곱의 기댓값 (독립인 경우만)

$X$와 $Y$가 **독립**이면:

$$E[XY] = E[X] \cdot E[Y]$$

**주의**: 이것은 일반적으로 종속 확률변수에 대해서는 성립하지 않습니다.

---

## 3. 분산

### 정의

$X$의 **분산**(variance)은 평균으로부터의 기대 제곱 편차를 측정합니다:

$$\text{Var}(X) = \sigma_X^2 = E\left[(X - \mu_X)^2\right]$$

### 간편 (계산) 공식

$$\text{Var}(X) = E[X^2] - (E[X])^2$$

*증명*:

$$E[(X - \mu)^2] = E[X^2 - 2\mu X + \mu^2] = E[X^2] - 2\mu E[X] + \mu^2 = E[X^2] - \mu^2$$

### 표준편차

$$\sigma_X = \sqrt{\text{Var}(X)}$$

표준편차는 $X$와 같은 단위를 가지므로 분산보다 해석이 용이합니다.

### 분산의 성질

1. $\text{Var}(X) \geq 0$ 항상 성립; $\text{Var}(X) = 0$이면 $X$가 거의 확실하게(a.s.) 상수
2. $\text{Var}(aX + b) = a^2 \, \text{Var}(X)$ (상수 이동; 스케일링은 제곱)
3. $X$와 $Y$가 **독립**이면: $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$
4. 일반적으로: $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)$

### 예제: 공정한 주사위의 분산

$E[X] = 3.5$이고 $E[X^2] = \frac{1}{6}(1 + 4 + 9 + 16 + 25 + 36) = \frac{91}{6} \approx 15.167$

$$\text{Var}(X) = E[X^2] - (E[X])^2 = \frac{91}{6} - \left(\frac{7}{2}\right)^2 = \frac{91}{6} - \frac{49}{4} = \frac{35}{12} \approx 2.917$$

---

## 4. 고차 적률: 왜도와 첨도

### $k$차 적률

$X$의 원점에 대한 **$k$차 적률**(moment):

$$\mu_k' = E[X^k]$$

**$k$차 중심 적률**(central moment):

$$\mu_k = E[(X - \mu)^k]$$

참고: $\mu_1 = 0$ 항상 성립, $\mu_2 = \text{Var}(X)$.

### 왜도

**왜도**(skewness)는 분포의 비대칭성을 측정합니다:

$$\gamma_1 = \frac{E[(X - \mu)^3]}{\sigma^3} = \frac{\mu_3}{\sigma^3}$$

- $\gamma_1 > 0$: 오른쪽 꼬리가 긴 분포 (right-skewed)
- $\gamma_1 = 0$: 대칭
- $\gamma_1 < 0$: 왼쪽 꼬리가 긴 분포 (left-skewed)

### 첨도

**첨도**(kurtosis)는 정규분포 대비 꼬리의 무거움을 측정합니다:

$$\gamma_2 = \frac{E[(X - \mu)^4]}{\sigma^4} = \frac{\mu_4}{\sigma^4}$$

정규분포의 첨도는 $\gamma_2 = 3$입니다. **초과 첨도**(excess kurtosis)는 $\gamma_2 - 3$:

- 초과 > 0 (**급첨**(leptokurtic)): 정규분포보다 꼬리가 무거움
- 초과 = 0 (**정첨**(mesokurtic)): 정규분포와 유사한 꼬리
- 초과 < 0 (**완첨**(platykurtic)): 정규분포보다 꼬리가 가벼움

---

## 5. 적률 생성 함수

### 정의

$X$의 **적률 생성 함수**(Moment Generating Function, MGF)는:

$$M_X(t) = E[e^{tX}]$$

0을 포함하는 어떤 열린 구간에서 모든 $t$에 대해 정의됩니다.

**이산**: $M_X(t) = \sum_{x} e^{tx} \, p_X(x)$

**연속**: $M_X(t) = \int_{-\infty}^{\infty} e^{tx} \, f_X(x) \, dx$

### 적률 추출

핵심 성질 -- 미분으로 적률을 추출할 수 있습니다:

$$E[X^k] = M_X^{(k)}(0) = \left.\frac{d^k}{dt^k} M_X(t)\right|_{t=0}$$

*증명 스케치*: $e^{tX} = \sum_{k=0}^{\infty} \frac{(tX)^k}{k!}$이므로 $M_X(t) = \sum_{k=0}^{\infty} \frac{t^k}{k!} E[X^k]$. $k$번 미분하고 $t = 0$을 대입하면 $E[X^k]$가 분리됩니다.

### 유일성 정리

두 확률변수가 $t = 0$의 근방에서 같은 MGF를 가지면, 같은 분포를 갖습니다. 이로 인해 MGF는 분포 식별에 강력한 도구가 됩니다.

### MGF의 성질

1. **상수**: $M_c(t) = e^{ct}$
2. **선형 변환**: $M_{aX+b}(t) = e^{bt} M_X(at)$
3. **독립 확률변수의 합**: $X \perp Y$이면, $M_{X+Y}(t) = M_X(t) \cdot M_Y(t)$

### 예제: Bernoulli($p$)의 MGF

$$M_X(t) = E[e^{tX}] = e^{t \cdot 0}(1-p) + e^{t \cdot 1}p = (1-p) + pe^t$$

1차 적률: $M_X'(t) = pe^t$이므로 $E[X] = M_X'(0) = p$.

2차 적률: $M_X''(t) = pe^t$이므로 $E[X^2] = M_X''(0) = p$.

분산: $\text{Var}(X) = p - p^2 = p(1-p)$.

---

## 6. 확률 부등식

### 마르코프 부등식

$X \geq 0$이고 $a > 0$이면:

$$P(X \geq a) \leq \frac{E[X]}{a}$$

*증명*: $E[X] = E[X \cdot \mathbf{1}_{X \geq a}] + E[X \cdot \mathbf{1}_{X < a}] \geq E[X \cdot \mathbf{1}_{X \geq a}] \geq a \cdot P(X \geq a)$.

**예제**: 분당 평균 서버 요청 수가 100이면, $P(X \geq 500) \leq 100/500 = 0.20$.

### 체비셰프 부등식

유한한 평균 $\mu$와 분산 $\sigma^2$을 갖는 임의의 확률변수 $X$에 대해, $k > 0$일 때:

$$P(|X - \mu| \geq k\sigma) \leq \frac{1}{k^2}$$

동치 형태:

$$P(|X - \mu| \geq a) \leq \frac{\sigma^2}{a^2}$$

*증명*: 비음 확률변수 $(X - \mu)^2$에 마르코프 부등식을 적용합니다.

**예제**: 공장에서 평균 길이 10 cm, 표준편차 0.1 cm인 볼트를 생산합니다. 볼트가 평균에서 0.3 cm 이상 벗어날 확률:

$$P(|X - 10| \geq 0.3) \leq \frac{(0.1)^2}{(0.3)^2} = \frac{1}{9} \approx 0.111$$

### 옌센 부등식

$g$가 **볼록**(convex) 함수 (즉, $g''(x) \geq 0$)이면:

$$E[g(X)] \geq g(E[X])$$

$g$가 **오목**(concave) ($g''(x) \leq 0$)이면, 부등호가 반대가 됩니다.

**예제**:

- $g(x) = x^2$는 볼록: $E[X^2] \geq (E[X])^2$ (즉, $\text{Var}(X) \geq 0$)
- $g(x) = \ln(x)$는 오목: $E[\ln X] \leq \ln(E[X])$ (정보 이론에서 사용)
- $g(x) = e^x$는 볼록: $E[e^X] \geq e^{E[X]}$

---

## 7. Python 예제

### PMF로부터 적률 계산

```python
def compute_moments(values, probs):
    """Compute mean, variance, skewness, and kurtosis from a PMF."""
    # Mean
    mean = sum(x * p for x, p in zip(values, probs))

    # Variance
    e_x2 = sum(x**2 * p for x, p in zip(values, probs))
    var = e_x2 - mean**2

    # Standard deviation
    std = var ** 0.5

    # Skewness
    mu3 = sum((x - mean)**3 * p for x, p in zip(values, probs))
    skew = mu3 / std**3 if std > 0 else 0

    # Kurtosis
    mu4 = sum((x - mean)**4 * p for x, p in zip(values, probs))
    kurt = mu4 / std**4 if std > 0 else 0
    excess_kurt = kurt - 3

    print(f"E[X]            = {mean:.4f}")
    print(f"Var(X)          = {var:.4f}")
    print(f"Std(X)          = {std:.4f}")
    print(f"Skewness        = {skew:.4f}")
    print(f"Kurtosis        = {kurt:.4f}")
    print(f"Excess Kurtosis = {excess_kurt:.4f}")
    return mean, var, skew, kurt

# Fair die
values = [1, 2, 3, 4, 5, 6]
probs = [1/6] * 6
compute_moments(values, probs)
# E[X] = 3.5, Var = 2.9167, Skew = 0, Excess Kurt = -1.2686
```

### 적률 생성 함수의 수치 계산

```python
import math

def mgf_bernoulli(t, p):
    """MGF of Bernoulli(p): (1-p) + p*e^t"""
    return (1 - p) + p * math.exp(t)

def numerical_derivative(f, x, h=1e-7):
    """Central difference approximation of f'(x)."""
    return (f(x + h) - f(x - h)) / (2 * h)

def numerical_second_derivative(f, x, h=1e-5):
    """Central difference approximation of f''(x)."""
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h)

p = 0.4
mgf = lambda t: mgf_bernoulli(t, p)

# E[X] = M'(0)
e_x = numerical_derivative(mgf, 0)
print(f"E[X] from MGF: {e_x:.6f}  (exact: {p})")

# E[X^2] = M''(0)
e_x2 = numerical_second_derivative(mgf, 0)
print(f"E[X^2] from MGF: {e_x2:.6f}  (exact: {p})")

# Var(X) = E[X^2] - (E[X])^2
var = e_x2 - e_x**2
print(f"Var(X) from MGF: {var:.6f}  (exact: {p*(1-p):.6f})")
```

### 기댓값의 선형성: 시뮬레이션

```python
import random

def demonstrate_linearity(n=500000):
    """Show E[2X + 3Y + 5] = 2E[X] + 3E[Y] + 5 even for dependent X, Y."""
    random.seed(42)
    sum_x = sum_y = sum_combo = 0

    for _ in range(n):
        x = random.gauss(0, 1)  # standard normal approximation
        y = x + random.gauss(0, 0.5)  # Y depends on X!
        z = 2 * x + 3 * y + 5

        sum_x += x
        sum_y += y
        sum_combo += z

    e_x = sum_x / n
    e_y = sum_y / n
    e_z = sum_combo / n
    expected = 2 * e_x + 3 * e_y + 5

    print(f"E[X]             = {e_x:.4f}")
    print(f"E[Y]             = {e_y:.4f}")
    print(f"E[2X + 3Y + 5]   = {e_z:.4f}")
    print(f"2E[X] + 3E[Y] + 5 = {expected:.4f}")
    print(f"Difference       = {abs(e_z - expected):.6f}")

demonstrate_linearity()
```

### 체비셰프 부등식 검증

```python
import random

def verify_chebyshev(n=1_000_000):
    """Verify Chebyshev's inequality with simulated data."""
    random.seed(99)
    # Exponential distribution: mean=1, var=1, std=1
    samples = [-math.log(1 - random.random()) for _ in range(n)]

    import math
    mean = sum(samples) / n
    var = sum((x - mean)**2 for x in samples) / (n - 1)
    std = math.sqrt(var)

    print(f"Sample mean = {mean:.4f}, Sample std = {std:.4f}\n")
    print(f"{'k':>4}  {'Chebyshev bound':>16}  {'Empirical P':>13}")
    print("-" * 38)
    for k in [1, 1.5, 2, 3, 4]:
        bound = 1 / k**2
        empirical = sum(1 for x in samples if abs(x - mean) >= k * std) / n
        print(f"{k:4.1f}  {bound:16.4f}  {empirical:13.4f}")

verify_chebyshev()
```

### 합의 분산: 독립 vs. 종속

```python
import random
import math

def variance_of_sum(n=500000):
    """Demonstrate Var(X+Y) = Var(X) + Var(Y) for independent X, Y."""
    random.seed(0)

    # Independent case
    xs = [random.gauss(2, 3) for _ in range(n)]
    ys = [random.gauss(5, 4) for _ in range(n)]
    sums_indep = [x + y for x, y in zip(xs, ys)]

    var_x = sum((x - 2)**2 for x in xs) / n
    var_y = sum((y - 5)**2 for y in ys) / n
    var_sum = sum((s - 7)**2 for s in sums_indep) / n

    print("=== Independent Case ===")
    print(f"Var(X)     = {var_x:.4f}  (theoretical: 9)")
    print(f"Var(Y)     = {var_y:.4f}  (theoretical: 16)")
    print(f"Var(X+Y)   = {var_sum:.4f}  (theoretical: 25)")
    print(f"Var(X)+Var(Y) = {var_x + var_y:.4f}\n")

    # Dependent case: Y = 2X + noise
    xs2 = [random.gauss(0, 1) for _ in range(n)]
    ys2 = [2 * x + random.gauss(0, 0.5) for x in xs2]
    sums_dep = [x + y for x, y in zip(xs2, ys2)]

    m_x = sum(xs2) / n
    m_y = sum(ys2) / n
    m_s = sum(sums_dep) / n
    var_x2 = sum((x - m_x)**2 for x in xs2) / n
    var_y2 = sum((y - m_y)**2 for y in ys2) / n
    var_sum2 = sum((s - m_s)**2 for s in sums_dep) / n

    print("=== Dependent Case (Y = 2X + noise) ===")
    print(f"Var(X)       = {var_x2:.4f}")
    print(f"Var(Y)       = {var_y2:.4f}")
    print(f"Var(X+Y)     = {var_sum2:.4f}")
    print(f"Var(X)+Var(Y) = {var_x2 + var_y2:.4f}")
    print(f"Var(X+Y) != Var(X)+Var(Y) when dependent!")

variance_of_sum()
```

---

## 8. 핵심 요약

1. **기댓값** $E[X]$는 모든 가능한 값의 확률 가중 평균입니다. 많은 반복에 걸친 장기적 평균을 나타냅니다.

2. **기댓값의 선형성** $E[aX + bY] = aE[X] + bE[Y]$는 종속성과 **무관하게** **항상** 성립합니다. 이것은 아마도 확률론에서 가장 유용한 성질입니다.

3. **분산** $\text{Var}(X) = E[X^2] - (E[X])^2$는 퍼짐을 측정합니다. 간편 공식이 거의 항상 계산하기 쉽습니다. 분산은 독립 (또는 비상관) 확률변수에 대해서만 가법적입니다.

4. **왜도**는 비대칭성을 측정하고, **첨도**는 정규분포 대비 꼬리의 무거움을 측정합니다.

5. **MGF** $M_X(t) = E[e^{tX}]$는 모든 적률을 담고 있습니다: $E[X^k] = M_X^{(k)}(0)$. 두 확률변수가 같은 MGF를 가지면 같은 분포를 갖습니다. 독립 확률변수 합의 MGF는 개별 MGF의 곱입니다.

6. **확률 부등식**은 분포와 무관한 한계를 제공합니다:
   - **마르코프**: $P(X \geq a) \leq E[X]/a$ ($X \geq 0$ 필요)
   - **체비셰프**: $P(|X - \mu| \geq k\sigma) \leq 1/k^2$ (평균과 분산만 필요)
   - **옌센**: 볼록 $g$에 대해 $E[g(X)] \geq g(E[X])$

---

*이전: [03 - 확률변수와 분포](./03_Random_Variables_and_Distributions.md) | 다음: [05 - 결합 분포](./05_Joint_Distributions.md)*
